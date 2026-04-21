from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.config import get_settings
from app.core.container import AppContainer
from app.evaluation.ragas_metrics import try_ragas_scores
from app.services import RAGService
from app.utils.mlflow_tracking import mlflow_run


async def run_evaluation(dataset_path: Path) -> list[dict[str, Any]]:
    settings = get_settings()
    container = AppContainer(settings)
    container.bootstrap_sample_data()
    service = RAGService(settings, container.retriever, container.generator, container.fallback)
    rows = _load_jsonl(dataset_path)
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            response = await service.answer(row["query"])
            result = {
                "query": row["query"],
                "reference": row.get("reference", ""),
                "answer": response.answer,
                "sources": [source.model_dump() for source in response.sources],
                "faithfulness_score": response.faithfulness_score,
                "confidence": response.confidence,
                "hallucination_detected": response.hallucination_detected,
                "fallback_triggered": response.fallback_triggered,
                "latency_ms": response.latency_ms,
            }
            results.append(result)
    finally:
        await container.close()
    _log_results(settings, results, dataset_path)
    return results


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _log_results(settings, results: list[dict[str, Any]], dataset_path: Path) -> None:
    if not results:
        return
    with mlflow_run(settings, "offline-evaluation"):
        try:
            import mlflow

            mlflow.log_param("dataset", str(dataset_path))
            mlflow.log_metric("mean_faithfulness", mean(r["faithfulness_score"] for r in results))
            mlflow.log_metric("mean_confidence", mean(r["confidence"] for r in results))
            mlflow.log_metric("mean_latency_ms", mean(r["latency_ms"] for r in results))
            mlflow.log_metric(
                "fallback_rate",
                sum(1 for r in results if r["fallback_triggered"]) / len(results),
            )
            if settings.enable_ragas:
                for key, value in try_ragas_scores(results).items():
                    mlflow.log_metric(key, value)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG hallucination detector.")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/eval_results.jsonl"))
    args = parser.parse_args()

    results = asyncio.run(run_evaluation(args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    print(json.dumps({"evaluated": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
