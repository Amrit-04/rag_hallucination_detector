from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Sentence-transformers wrapper with an in-process query cache."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    @lru_cache(maxsize=4096)
    def embed_query_tuple(self, text: str) -> tuple[float, ...]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return tuple(float(value) for value in embedding)

    def embed_query(self, text: str) -> list[float]:
        return list(self.embed_query_tuple(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings.astype(float).tolist()


def cosine_similarity(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom == 0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / denom)

