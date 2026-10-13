"""FastAPI application entrypoint for Filewise."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.services.embeddings import EmbeddingService
from app.services.llm_provider import create_llm_provider
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("filewise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize services and dependencies once upon startup."""
    logger.info("Initializing Filewise backend services...")

    # Initialize Vector Store if not already injected
    if not hasattr(app.state, "vector_store") or app.state.vector_store is None:
        app.state.vector_store = VectorStore(storage_dir=settings.vector_store_path)

    # Initialize Embedding Service if not already injected
    if not hasattr(app.state, "embedding_service") or app.state.embedding_service is None:
        app.state.embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)

    # Initialize Retrieval Service if not already injected
    if not hasattr(app.state, "retrieval_service") or app.state.retrieval_service is None:
        app.state.retrieval_service = RetrievalService(
            vector_store=app.state.vector_store,
            embedding_service=app.state.embedding_service,
        )

    # Initialize LLM Provider if not already injected
    if not hasattr(app.state, "llm_provider") or app.state.llm_provider is None:
        app.state.llm_provider = create_llm_provider(
            provider_type=settings.LLM_PROVIDER,
            api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
        )

    # Initialize QA Service if not already injected
    if not hasattr(app.state, "qa_service") or app.state.qa_service is None:
        app.state.qa_service = QAService(
            retrieval_service=app.state.retrieval_service,
            llm_provider=app.state.llm_provider,
            default_top_k=settings.RETRIEVAL_TOP_K,
            default_threshold=settings.RELEVANCE_THRESHOLD,
        )
    logger.info(
        "Filewise services initialized successfully (LLM=%s, Embeddings=%s, Store=%s).",
        settings.LLM_PROVIDER,
        settings.EMBEDDING_MODEL_NAME,
        settings.vector_store_path,
    )
    yield
    logger.info("Shutting down Filewise...")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    app = FastAPI(
        title="Filewise — Evidence-Based Knowledge Assistant",
        description="Grounded AI knowledge assistant answering questions strictly from uploaded documents.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes with both /api/v1 prefix and root level endpoints
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(api_router)  # Also supports /health, /documents, /documents/ingest, /ask directly

    # Static UI route
    static_dir = Path(__file__).resolve().parent / "static"
    if (static_dir / "index.html").exists():
        @app.get("/", include_in_schema=False)
        async def serve_ui():
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
