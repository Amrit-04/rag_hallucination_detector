from __future__ import annotations

from app.core.config import Settings
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.types import Document, ScoredDocument
from app.retrieval.vector_store import ChromaVectorStore


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        vector_store: ChromaVectorStore,
        sparse_retriever: BM25Retriever,
        reranker: CrossEncoderReranker,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker

    def add_documents(self, documents: list[Document]) -> None:
        self.vector_store.add(documents)
        self.sparse_retriever.add(documents)

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredDocument]:
        dense = self.vector_store.search(query, self.settings.retrieval_dense_k)
        sparse = self.sparse_retriever.search(query, self.settings.retrieval_sparse_k)
        fused = self._weighted_fusion(dense, sparse)
        rerank_candidates = fused[: self.settings.rerank_top_k]
        final_k = top_k or self.settings.retrieval_final_k
        return self.reranker.rerank(query, rerank_candidates, final_k)

    def count(self) -> int:
        return self.vector_store.count()

    def _weighted_fusion(
        self,
        dense_results: list[ScoredDocument],
        sparse_results: list[ScoredDocument],
    ) -> list[ScoredDocument]:
        by_id: dict[str, ScoredDocument] = {}
        for doc in dense_results:
            by_id[doc.id] = doc
        for doc in sparse_results:
            existing = by_id.get(doc.id)
            if existing:
                existing.sparse_score = doc.sparse_score
                existing.metadata = {**doc.metadata, **existing.metadata}
            else:
                by_id[doc.id] = doc

        for doc in by_id.values():
            doc.score = (
                self.settings.dense_weight * doc.dense_score
                + self.settings.sparse_weight * doc.sparse_score
            )
        return sorted(by_id.values(), key=lambda item: item.score, reverse=True)

