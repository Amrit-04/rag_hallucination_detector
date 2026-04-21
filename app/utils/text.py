import re
from hashlib import sha256


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def stable_id(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:24]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split long documents into overlapping chunks.

    Character-based chunking is deterministic and dependency-light; callers can
    replace it with LangChain splitters without changing the retrieval contract.
    """

    cleaned = normalize_whitespace(text)
    if len(cleaned) <= chunk_size:
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]

