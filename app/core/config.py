from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven application settings.

    The defaults are chosen so a developer can run the service locally with
    Ollama and a small on-disk Chroma collection without extra infrastructure.
    """

    app_name: str = "rag-hallucination-detector"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: float = 60.0
    generation_temperature: float = 0.1
    generation_max_tokens: int = 512

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chroma_persist_dir: Path = Path("models/chroma")
    chroma_collection_name: str = "rag_documents"
    document_store_path: Path = Path("models/document_store.jsonl")
    ingestion_batch_size: int = 64

    dense_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    retrieval_dense_k: int = 12
    retrieval_sparse_k: int = 12
    retrieval_final_k: int = 5
    rerank_top_k: int = 8

    faithfulness_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.38, ge=0.0, le=1.0)
    verification_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    fallback_strategy: Literal["regenerate", "abstain", "snippets"] = "regenerate"
    allow_second_fallback: bool = True

    enable_mlflow: bool = True
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "rag-hallucination-detector"

    enable_ragas: bool = True
    request_timeout_seconds: float = 65.0

    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

