from __future__ import annotations

from app.core.config import Settings
from app.detection.scorer import DetectionResult, HallucinationDetector
from app.generation.service import GenerationService
from app.retrieval.types import ScoredDocument


class FallbackController:
    def __init__(
        self,
        settings: Settings,
        generator: GenerationService,
        detector: HallucinationDetector,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.detector = detector

    async def apply(
        self,
        query: str,
        answer: str,
        documents: list[ScoredDocument],
        detection: DetectionResult,
    ) -> tuple[str, DetectionResult, bool]:
        if not detection.hallucination_detected:
            return answer, detection, False

        if self.settings.fallback_strategy == "abstain":
            return self._abstain(), detection, True

        if self.settings.fallback_strategy == "snippets":
            return self._source_snippets(documents), detection, True

        regenerated = await self.generator.generate(query, documents, strict=True)
        regenerated_detection = self.detector.score(query, regenerated, documents)
        if regenerated_detection.hallucination_detected and self.settings.allow_second_fallback:
            return self._abstain(), regenerated_detection, True
        return regenerated, regenerated_detection, True

    @staticmethod
    def _abstain() -> str:
        return "I don't have enough information in the retrieved sources to answer that."

    @staticmethod
    def _source_snippets(documents: list[ScoredDocument]) -> str:
        if not documents:
            return FallbackController._abstain()
        snippets = [f"- {doc.text[:450].strip()}" for doc in documents[:3]]
        return "Relevant source snippets:\n" + "\n".join(snippets)

