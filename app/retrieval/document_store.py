from __future__ import annotations

import json
from pathlib import Path

from app.retrieval.types import Document


class JsonlDocumentStore:
    """Small durable store used by BM25 and source reconstruction.

    Chroma owns dense vectors. This file keeps raw text and metadata in a simple
    append-friendly format so the sparse index can be rebuilt quickly at startup.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[Document]:
        if not self.path.exists():
            return []
        documents: list[Document] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                documents.append(
                    Document(
                        id=item["id"],
                        text=item["text"],
                        metadata=item.get("metadata", {}),
                    )
                )
        return documents

    def append_many(self, documents: list[Document]) -> None:
        if not documents:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for doc in documents:
                handle.write(
                    json.dumps(
                        {"id": doc.id, "text": doc.text, "metadata": doc.metadata},
                        ensure_ascii=True,
                    )
                    + "\n"
                )

