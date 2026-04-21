from __future__ import annotations

import math
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.core.config import Settings
from app.retrieval.embeddings import EmbeddingService, cosine_similarity
from app.retrieval.types import ScoredDocument
from app.utils.text import split_sentences


@dataclass(slots=True)
class DetectionResult:
    faithfulness_score: float
    semantic_similarity: float
    sentence_support: float
    verification_score: float
    hallucination_detected: bool
    confidence: float

    def as_details(self) -> dict[str, float]:
        return {
            "semantic_similarity": self.semantic_similarity,
            "sentence_support": self.sentence_support,
            "verification_score": self.verification_score,
        }


class HallucinationDetector:
    """Post-generation grounding checks.

    This intentionally combines cheap semantic checks with a stronger
    cross-encoder verification signal. The scores are calibrated into [0, 1] so
    product teams can tune a single faithfulness threshold per deployment.
    """

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        verifier_model_name: str,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.verifier = CrossEncoder(verifier_model_name)

    def score(self, query: str, answer: str, documents: list[ScoredDocument]) -> DetectionResult:
        if not answer.strip() or not documents:
            return DetectionResult(0.0, 0.0, 0.0, 0.0, True, 0.0)

        semantic_similarity = self._answer_context_similarity(answer, documents)
        sentence_support = self._sentence_support(answer, documents)
        verification_score = self._cross_encoder_verification(query, answer, documents)

        faithfulness = (
            0.35 * semantic_similarity
            + 0.35 * sentence_support
            + 0.30 * verification_score
        )
        faithfulness = max(0.0, min(1.0, faithfulness))
        hallucination = (
            faithfulness < self.settings.faithfulness_threshold
            or semantic_similarity < self.settings.similarity_threshold
            or verification_score < self.settings.verification_threshold
        )
        confidence = max(0.0, min(1.0, faithfulness * (0.7 + 0.3 * max((d.score for d in documents), default=0.0))))

        return DetectionResult(
            faithfulness_score=faithfulness,
            semantic_similarity=semantic_similarity,
            sentence_support=sentence_support,
            verification_score=verification_score,
            hallucination_detected=hallucination,
            confidence=confidence,
        )

    def _answer_context_similarity(self, answer: str, documents: list[ScoredDocument]) -> float:
        answer_embedding = self.embeddings.embed_query(answer)
        similarities = [
            cosine_similarity(answer_embedding, self.embeddings.embed_query(doc.text))
            for doc in documents
        ]
        if not similarities:
            return 0.0
        return max(0.0, min(1.0, float(sum(sorted(similarities, reverse=True)[:3]) / min(3, len(similarities)))))

    def _sentence_support(self, answer: str, documents: list[ScoredDocument]) -> float:
        sentences = split_sentences(answer)
        if not sentences:
            return 0.0
        context_embeddings = [self.embeddings.embed_query(doc.text) for doc in documents]
        supported = 0
        for sentence in sentences:
            sentence_embedding = self.embeddings.embed_query(sentence)
            best = max(cosine_similarity(sentence_embedding, ctx) for ctx in context_embeddings)
            if best >= self.settings.similarity_threshold:
                supported += 1
        return supported / len(sentences)

    def _cross_encoder_verification(
        self,
        query: str,
        answer: str,
        documents: list[ScoredDocument],
    ) -> float:
        context = "\n\n".join(doc.text for doc in documents[:3])
        raw = float(self.verifier.predict([(f"{query}\n{answer}", context)])[0])
        return 1.0 / (1.0 + math.exp(-raw))

