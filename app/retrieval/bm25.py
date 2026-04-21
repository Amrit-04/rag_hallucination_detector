from __future__ import annotations

import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from app.retrieval.types import Document, ScoredDocument

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class BM25Retriever:
    def __init__(self, documents: Iterable[Document] | None = None) -> None:
        self.documents: list[Document] = []
        self.index: BM25Okapi | None = None
        self.rebuild(list(documents or []))

    def rebuild(self, documents: list[Document]) -> None:
        self.documents = documents
        tokenized = [tokenize(doc.text) for doc in self.documents]
        self.index = BM25Okapi(tokenized) if tokenized else None

    def add(self, documents: list[Document]) -> None:
        self.rebuild([*self.documents, *documents])

    def search(self, query: str, k: int) -> list[ScoredDocument]:
        if not self.index or not self.documents:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        max_score = max((float(score) for _, score in ranked), default=0.0)
        results: list[ScoredDocument] = []
        for idx, raw_score in ranked:
            doc = self.documents[idx]
            normalized = float(raw_score / max_score) if max_score > 0 else 0.0
            results.append(
                ScoredDocument(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=normalized,
                    sparse_score=normalized,
                )
            )
        return results

