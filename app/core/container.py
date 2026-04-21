from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import Settings
from app.detection.fallback import FallbackController
from app.detection.scorer import HallucinationDetector
from app.generation.ollama_client import OllamaClient
from app.generation.service import GenerationService
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.document_store import JsonlDocumentStore
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ingestion import IngestionService
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class AppContainer:
    """Dependency container created once at FastAPI startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.document_store = JsonlDocumentStore(settings.document_store_path)
        self.embeddings = EmbeddingService(settings.embedding_model_name)
        self.vector_store = ChromaVectorStore(
            settings.chroma_persist_dir,
            settings.chroma_collection_name,
            self.embeddings,
        )
        existing_docs = self.document_store.load_all()
        self.bm25 = BM25Retriever(existing_docs)
        self.reranker = CrossEncoderReranker(settings.cross_encoder_model_name)
        self.retriever = HybridRetriever(settings, self.vector_store, self.bm25, self.reranker)
        self.ingestion = IngestionService(self.retriever, self.document_store)
        self.ollama = OllamaClient(settings)
        self.generator = GenerationService(self.ollama)
        self.detector = HallucinationDetector(
            settings,
            self.embeddings,
            settings.cross_encoder_model_name,
        )
        self.fallback = FallbackController(settings, self.generator, self.detector)

    async def close(self) -> None:
        await self.ollama.close()

    def bootstrap_sample_data(self, path: Path = Path("data/sample_docs.jsonl")) -> None:
        if self.retriever.count() > 0 or not path.exists():
            return
        logger.info("Bootstrapping sample documents from %s", path)
        items = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    items.append(json.loads(line))
        self.ingestion.ingest_raw_documents(items)

