from __future__ import annotations

import time

from app.core.config import Settings
from app.detection.fallback import FallbackController
from app.generation.service import GenerationService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import ScoredDocument
from app.schemas import QueryResponse, Source
from app.utils.mlflow_tracking import log_query_metrics


class RAGService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: GenerationService,
        fallback: FallbackController,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.fallback = fallback

    async def answer(self, query: str, top_k: int | None = None) -> QueryResponse:
        started = time.perf_counter()
        documents = self.retriever.retrieve(query, top_k)
        answer = await self.generator.generate(query, documents)
        detection = self.fallback.detector.score(query, answer, documents)
        final_answer, final_detection, fallback_triggered = await self.fallback.apply(
            query,
            answer,
            documents,
            detection,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        response = QueryResponse(
            answer=final_answer,
            sources=self._sources(documents),
            faithfulness_score=final_detection.faithfulness_score,
            hallucination_detected=final_detection.hallucination_detected,
            fallback_triggered=fallback_triggered,
            latency_ms=latency_ms,
            confidence=final_detection.confidence,
            detection_details=final_detection.as_details(),
        )
        log_query_metrics(
            self.settings,
            {
                "faithfulness_score": response.faithfulness_score,
                "confidence": response.confidence,
                "latency_ms": float(response.latency_ms),
            },
            {
                "model": self.settings.ollama_model,
                "fallback_triggered": str(response.fallback_triggered),
            },
        )
        return response

    @staticmethod
    def _sources(documents: list[ScoredDocument]) -> list[Source]:
        return [
            Source(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=doc.score,
            )
            for doc in documents
        ]

