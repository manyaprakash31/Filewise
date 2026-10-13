"""Pydantic schemas for request and response validation in Filewise."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete chunk of text extracted from a document."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk, e.g. docname_000")
    document_name: str = Field(..., description="Sanitized source document filename")
    page_number: Optional[int] = Field(None, description="1-based page number if available (e.g., from PDF)")
    text: str = Field(..., description="Text content of the chunk")


class SourceCitation(BaseModel):
    """Citation metadata and verbatim excerpt supporting an answer."""

    document: str = Field(..., description="Original document filename")
    page: Optional[int] = Field(None, description="Page number where the evidence appears")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")
    excerpt: str = Field(..., description="Verbatim excerpt from the retrieved document chunk")
    relevance_score: Optional[float] = Field(None, description="Cosine similarity score of chunk to question")


class IngestResponse(BaseModel):
    """Response returned after ingesting a single document."""

    filename: str
    status: str = Field(..., description="Status string: 'indexed', 'already_indexed', or 'failed'")
    chunks_indexed: int
    message: str


class BatchIngestResponse(BaseModel):
    """Response returned after ingesting multiple documents."""

    total_files: int
    successful: int
    failed: int
    results: List[IngestResponse]


class IndexedDocument(BaseModel):
    """Metadata summary of an indexed document in the vector store."""

    filename: str
    chunk_count: int
    page_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response listing all currently indexed documents."""

    total_documents: int
    documents: List[IndexedDocument]


class AskRequest(BaseModel):
    """Question answering query request."""

    question: str = Field(..., min_length=2, max_length=1000, description="The user's query question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Optional override for top_k retrieval")
    relevance_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Optional override for similarity score threshold"
    )


class AskResponse(BaseModel):
    """Evidence-grounded response to a user query."""

    question: str
    answer: str
    answerable: bool
    sources: List[SourceCitation] = Field(default_factory=list)
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = "healthy"
    app: str = "Filewise — Evidence-Based Knowledge Assistant"
    version: str = "1.0.0"
    llm_provider: str
    embedding_model: str
    indexed_documents_count: int
    vector_store_ready: bool
