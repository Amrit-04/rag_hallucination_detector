from __future__ import annotations

from app.retrieval.document_store import JsonlDocumentStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import Document
from app.utils.text import chunk_text, stable_id


class IngestionService:
    def __init__(self, retriever: HybridRetriever, document_store: JsonlDocumentStore) -> None:
        self.retriever = retriever
        self.document_store = document_store

    def ingest_raw_documents(self, items: list[dict]) -> int:
        documents: list[Document] = []
        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {})
            parent_id = item.get("id") or stable_id(text)
            for idx, chunk in enumerate(chunk_text(text)):
                doc_id = f"{parent_id}:{idx}"
                documents.append(
                    Document(
                        id=doc_id,
                        text=chunk,
                        metadata={**metadata, "parent_id": parent_id, "chunk": idx},
                    )
                )
        self.retriever.add_documents(documents)
        self.document_store.append_many(documents)
        return len(documents)

