from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredDocument(Document):
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0

