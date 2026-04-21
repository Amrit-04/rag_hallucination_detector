from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from app.retrieval.types import ScoredDocument

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        logger.info("Loading cross-encoder model: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[ScoredDocument], top_k: int) -> list[ScoredDocument]:
        if not documents:
            return []
        pairs = [(query, doc.text) for doc in documents]
        raw_scores = self.model.predict(pairs)
        min_score = float(min(raw_scores))
        max_score = float(max(raw_scores))
        spread = max(max_score - min_score, 1e-6)
        reranked: list[ScoredDocument] = []
        for doc, raw_score in zip(documents, raw_scores, strict=False):
            normalized = (float(raw_score) - min_score) / spread
            doc.rerank_score = normalized
            doc.score = (doc.score * 0.35) + (normalized * 0.65)
            reranked.append(doc)
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

