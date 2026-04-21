from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int | None = Field(default=None, ge=1, le=20)
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    faithfulness_score: float
    hallucination_detected: bool
    fallback_triggered: bool
    latency_ms: int
    confidence: float
    detection_details: dict[str, float] = Field(default_factory=dict)


class IngestDocument(BaseModel):
    id: str | None = None
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested: int
    collection: str


class HealthResponse(BaseModel):
    status: str
    collection_count: int

