"""FastAPI route definitions for Filewise."""

import logging
from typing import List
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from app.core.config import settings
from app.models.schemas import (
    AskRequest,
    AskResponse,
    BatchIngestResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
)
from app.services.chunking import TextChunker
from app.services.document_loader import DocumentLoader

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Application Health Check")
async def health(request: Request) -> HealthResponse:
    """Return health status, active LLM provider, embedding model, and vector store stats."""
    vector_store = request.app.state.vector_store
    doc_count = len(vector_store.list_documents())
    return HealthResponse(
        status="healthy",
        app="Filewise — Evidence-Based Knowledge Assistant",
        version="1.0.0",
        llm_provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        indexed_documents_count=doc_count,
        vector_store_ready=True,
    )


@router.post("/documents/ingest", response_model=IngestResponse, summary="Ingest Single Document")
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="PDF, TXT, or Markdown document"),
) -> IngestResponse:
    """Validate, parse, chunk, embed, and index a single document."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file missing filename.")

    sanitized_name = DocumentLoader.sanitize_filename(file.filename)

    if not DocumentLoader.is_supported(sanitized_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension for '{file.filename}'. Supported formats: PDF, TXT, Markdown (.md).",
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size limit of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    vector_store = request.app.state.vector_store
    embedding_service = request.app.state.embedding_service

    # Prevent duplicate indexing
    if vector_store.is_document_indexed(sanitized_name):
        return IngestResponse(
            filename=sanitized_name,
            status="already_indexed",
            chunks_indexed=0,
            message=f"Document '{sanitized_name}' has already been indexed. Duplicate avoided.",
        )

    # Parse document
    try:
        pages = DocumentLoader.load_bytes(sanitized_name, content)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error parsing %s: %s", sanitized_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing document content: {str(e)}",
        )

    # Chunk text
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    chunks = chunker.chunk_document(sanitized_name, pages)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text could not be divided into searchable chunks.",
        )

    # Generate embeddings and store
    try:
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)
        vector_store.add_chunks(chunks, embeddings)
    except Exception as e:
        logger.error("Error generating embeddings or indexing for %s: %s", sanitized_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings or save to vector store.",
        )

    return IngestResponse(
        filename=sanitized_name,
        status="indexed",
        chunks_indexed=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{sanitized_name}'.",
    )


@router.post("/documents/ingest/batch", response_model=BatchIngestResponse, summary="Batch Ingest Documents")
async def ingest_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="List of PDF, TXT, or Markdown documents"),
) -> BatchIngestResponse:
    """Accept multiple files, indexing each and returning granular results without failing the whole batch."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided in batch upload.")

    results: List[IngestResponse] = []
    vector_store = request.app.state.vector_store
    embedding_service = request.app.state.embedding_service
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    successful = 0
    failed = 0

    for file in files:
        raw_name = file.filename or "unknown"
        sanitized_name = DocumentLoader.sanitize_filename(raw_name)

        if not DocumentLoader.is_supported(sanitized_name):
            results.append(
                IngestResponse(
                    filename=raw_name,
                    status="failed",
                    chunks_indexed=0,
                    message="Unsupported file format.",
                )
            )
            failed += 1
            continue

        try:
            content = await file.read()
            if len(content) > settings.max_upload_size_bytes:
                results.append(
                    IngestResponse(
                        filename=sanitized_name,
                        status="failed",
                        chunks_indexed=0,
                        message=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                    )
                )
                failed += 1
                continue

            if vector_store.is_document_indexed(sanitized_name):
                results.append(
                    IngestResponse(
                        filename=sanitized_name,
                        status="already_indexed",
                        chunks_indexed=0,
                        message=f"Document '{sanitized_name}' already indexed.",
                    )
                )
                successful += 1
                continue

            pages = DocumentLoader.load_bytes(sanitized_name, content)
            chunks = chunker.chunk_document(sanitized_name, pages)
            if not chunks:
                results.append(
                    IngestResponse(
                        filename=sanitized_name,
                        status="failed",
                        chunks_indexed=0,
                        message="No extractable chunks found.",
                    )
                )
                failed += 1
                continue

            chunk_texts = [c.text for c in chunks]
            embeddings = embedding_service.embed_texts(chunk_texts)
            vector_store.add_chunks(chunks, embeddings)

            results.append(
                IngestResponse(
                    filename=sanitized_name,
                    status="indexed",
                    chunks_indexed=len(chunks),
                    message=f"Indexed {len(chunks)} chunks.",
                )
            )
            successful += 1
        except Exception as e:
            logger.warning("Batch upload failed for %s: %s", raw_name, e)
            results.append(
                IngestResponse(
                    filename=sanitized_name,
                    status="failed",
                    chunks_indexed=0,
                    message=f"Processing failed: {str(e)}",
                )
            )
            failed += 1

    return BatchIngestResponse(
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results,
    )


@router.get("/documents", response_model=DocumentListResponse, summary="List Indexed Documents")
async def list_documents(request: Request) -> DocumentListResponse:
    """Return list of all documents currently indexed in the vector store."""
    vector_store = request.app.state.vector_store
    docs = vector_store.list_documents()
    return DocumentListResponse(total_documents=len(docs), documents=docs)


@router.post("/ask", response_model=AskResponse, summary="Ask Question Grounded in Documents")
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """Retrieve relevant document chunks, query LLM, and return grounded answer with source citations."""
    qa_service = request.app.state.qa_service
    try:
        response = qa_service.answer_question(body)
        return response
    except Exception as e:
        logger.error("Error answering question '%s': %s", body.question, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering the question: {str(e)}",
        )
