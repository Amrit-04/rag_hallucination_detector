from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def try_ragas_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute RAGAS faithfulness and answer relevancy when configured.

    RAGAS APIs and local LLM wrappers evolve quickly, so this helper is isolated
    from the main evaluation path. If a deployment has compatible RAGAS and LLM
    configuration, these metrics are logged; otherwise local detector metrics
    remain the source of truth.
    """

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        dataset = Dataset.from_list(
            [
                {
                    "question": row["query"],
                    "answer": row["answer"],
                    "contexts": [source["text"] for source in row.get("sources", [])],
                    "ground_truth": row.get("reference", ""),
                }
                for row in rows
            ]
        )
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        dataframe = result.to_pandas()
        return {
            "ragas_faithfulness": float(dataframe["faithfulness"].mean()),
            "ragas_answer_relevancy": float(dataframe["answer_relevancy"].mean()),
        }
    except Exception as exc:
        logger.warning("RAGAS metrics skipped: %s", exc)
        return {}

