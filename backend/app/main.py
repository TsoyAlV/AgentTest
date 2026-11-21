from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .auth import current_user
from .config import Settings
from .db import Retriever
from .embeddings import Embeddings
from .errors import ServiceError
from .generation import Generator
from .schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from .chat_store import ChatStore
from .chat_api import ChatService, router as chat_router
from .memory import ConversationMemory


def create_app(settings=None, services=None):
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        if services is not None:
            app.state.services = services
            if hasattr(services,'chat_store'):
                app.state.chat_service=ChatService(services,services.chat_store,cfg.max_concurrent_generations,
                                                  ConversationMemory(cfg,services.generator))
            yield
            return
        retriever = Retriever(cfg)
        with httpx.Client(timeout=cfg.http_timeout_seconds, trust_env=False) as client:
            retriever.open()
            app.state.services = SimpleNamespace(
                embeddings=Embeddings(cfg, client), retriever=retriever,
                generator=Generator(cfg, client),
            )
            app.state.chat_service=ChatService(app.state.services,ChatStore(retriever),cfg.max_concurrent_generations,
                                              ConversationMemory(cfg,app.state.services.generator))
            try:
                yield
            finally:
                retriever.close()

    app = FastAPI(title="Project RAG API", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg
    app.include_router(chat_router)
    static=Path(__file__).parent/'static'
    app.mount('/static',StaticFiles(directory=static),name='static')

    @app.get('/',include_in_schema=False)
    def web_chat():return FileResponse(static/'index.html')

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": cfg.app_env,
                "embeddings_configured": bool(cfg.embedding_base_url)}

    def retrieve(request, user_id, query, limit):
        svc = request.app.state.services
        vector = svc.embeddings.encode(query)
        return svc.retriever.search(user_id, query, vector, limit)

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request, user_id: int = Depends(current_user)):
        return SearchResponse(results=retrieve(request, user_id, body.query, body.limit))

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request, user_id: int = Depends(current_user)):
        chunks = retrieve(request, user_id, body.question, body.limit)
        return request.app.state.services.generator.answer(body.question, chunks)

    @app.post("/ask/stream", response_class=StreamingResponse,
              responses={200: {"content": {"text/event-stream": {}},
                               "description": "SSE: sources, delta, done or error"}})
    def ask_stream(body: AskRequest, request: Request, user_id: int = Depends(current_user)):
        # Retrieval completes and releases its database connection before streaming.
        chunks = retrieve(request, user_id, body.question, body.limit)

        def events():
            try:
                for event, data in request.app.state.services.generator.stream_answer(body.question, chunks):
                    yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            except ServiceError as exc:
                # Headers are already sent; report upstream failures inside the stream.
                yield f"event: error\ndata: {json.dumps({'detail': str(exc), 'status': exc.status_code})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    return app
