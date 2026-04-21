from __future__ import annotations

from app.generation.ollama_client import OllamaClient
from app.generation.prompts import GROUNDED_ANSWER_PROMPT, STRICT_REGENERATION_PROMPT
from app.retrieval.types import ScoredDocument


class GenerationService:
    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    async def generate(self, query: str, documents: list[ScoredDocument], strict: bool = False) -> str:
        prompt_template = STRICT_REGENERATION_PROMPT if strict else GROUNDED_ANSWER_PROMPT
        prompt = prompt_template.format(query=query, context=self._format_context(documents))
        return await self.ollama.generate(prompt)

    def build_prompt(self, query: str, documents: list[ScoredDocument], strict: bool = False) -> str:
        prompt_template = STRICT_REGENERATION_PROMPT if strict else GROUNDED_ANSWER_PROMPT
        return prompt_template.format(query=query, context=self._format_context(documents))

    @staticmethod
    def _format_context(documents: list[ScoredDocument]) -> str:
        return "\n\n".join(
            f"[Source {idx + 1} | id={doc.id} | score={doc.score:.3f}]\n{doc.text}"
            for idx, doc in enumerate(documents)
        )

