from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.container import AppContainer
from app.schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.services import RAGService

router = APIRouter()


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_rag_service(container: AppContainer = Depends(get_container)) -> RAGService:
    return RAGService(container.settings, container.retriever, container.generator, container.fallback)


@router.get("/health", response_model=HealthResponse)
async def health(container: AppContainer = Depends(get_container)) -> HealthResponse:
    return HealthResponse(status="ok", collection_count=container.retriever.count())


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    payload: IngestRequest,
    container: AppContainer = Depends(get_container),
) -> IngestResponse:
    count = container.ingestion.ingest_raw_documents([item.model_dump() for item in payload.documents])
    return IngestResponse(ingested=count, collection=container.settings.chroma_collection_name)


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    service: RAGService = Depends(get_rag_service),
    container: AppContainer = Depends(get_container),
):
    if container.retriever.count() == 0:
        raise HTTPException(status_code=409, detail="No documents indexed. Call /ingest first.")

    if payload.stream:
        documents = container.retriever.retrieve(payload.query, payload.top_k)
        prompt = container.generator.build_prompt(payload.query, documents)

        async def event_stream():
            async for token_json in container.ollama.stream_generate(prompt):
                if await request.is_disconnected():
                    break
                try:
                    token = json.loads(token_json).get("response", "")
                except json.JSONDecodeError:
                    token = ""
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return await service.answer(payload.query, payload.top_k)

