from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.types import Document, ScoredDocument


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embeddings: EmbeddingService,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        vectors = self.embeddings.embed_documents([doc.text for doc in documents])
        self.collection.upsert(
            ids=[doc.id for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=vectors,
        )

    def search(self, query: str, k: int) -> list[ScoredDocument]:
        if self.count() == 0:
            return []
        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        scored: list[ScoredDocument] = []
        for doc_id, text, metadata, distance in zip(ids, docs, metadatas, distances, strict=False):
            score = max(0.0, 1.0 - float(distance))
            scored.append(
                ScoredDocument(
                    id=doc_id,
                    text=text,
                    metadata=metadata or {},
                    score=score,
                    dense_score=score,
                )
            )
        return scored

    def count(self) -> int:
        return int(self.collection.count())

