from __future__ import annotations

import httpx

from app.core.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout_seconds,
        )

    async def generate(self, prompt: str) -> str:
        response = await self.client.post(
            "/api/generate",
            json={
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.settings.generation_temperature,
                    "num_predict": self.settings.generation_max_tokens,
                },
            },
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()

    async def stream_generate(self, prompt: str):
        async with self.client.stream(
            "POST",
            "/api/generate",
            json={
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.settings.generation_temperature,
                    "num_predict": self.settings.generation_max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield line

    async def close(self) -> None:
        await self.client.aclose()

