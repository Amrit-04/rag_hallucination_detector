from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.container import AppContainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL documents into Chroma and BM25.")
    parser.add_argument("--input", type=Path, default=Path("data/sample_docs.jsonl"))
    args = parser.parse_args()

    settings = get_settings()
    container = AppContainer(settings)
    with args.input.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    count = container.ingestion.ingest_raw_documents(rows)
    print(json.dumps({"ingested_chunks": count, "collection": settings.chroma_collection_name}))


if __name__ == "__main__":
    main()

