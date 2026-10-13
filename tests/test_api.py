"""Integration tests for FastAPI endpoints."""

import io
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import create_app
from app.models.schemas import DocumentChunk
from app.services.embeddings import EmbeddingService
from app.services.llm_provider import MockLLMProvider
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore
from tests.test_retrieval import DummyEmbeddingService


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_store = VectorStore(storage_dir=Path(tmpdir))
        test_embeddings = DummyEmbeddingService()
        test_retrieval = RetrievalService(test_store, test_embeddings)
        test_llm = MockLLMProvider()
        test_qa = QAService(test_retrieval, test_llm)

        app = create_app()
        # Override state dependencies for fast isolated testing
        app.state.vector_store = test_store
        app.state.embedding_service = test_embeddings
        app.state.retrieval_service = test_retrieval
        app.state.llm_provider = test_llm
        app.state.qa_service = test_qa

        with TestClient(app) as test_client:
            yield test_client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == "Filewise — Evidence-Based Knowledge Assistant"
    assert data["vector_store_ready"] is True


def test_documents_empty_list(client):
    res = client.get("/documents")
    assert res.status_code == 200
    data = res.json()
    assert data["total_documents"] == 0
    assert data["documents"] == []


def test_ingest_single_document(client):
    file_content = b"The remote home office stipend is $1200 upon hire."
    files = {"file": ("policy.txt", io.BytesIO(file_content), "text/plain")}

    res = client.post("/documents/ingest", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "policy.txt"
    assert data["status"] == "indexed"
    assert data["chunks_indexed"] > 0

    # Test duplicate prevention
    res_dup = client.post("/documents/ingest", files={"file": ("policy.txt", io.BytesIO(file_content), "text/plain")})
    assert res_dup.status_code == 200
    assert res_dup.json()["status"] == "already_indexed"


def test_ask_question_unanswerable_empty_store(client):
    res = client.post("/ask", json={"question": "What is the capital of France?"})
    assert res.status_code == 200
    data = res.json()
    assert data["answerable"] is False
    assert "could not find sufficient information" in data["answer"].lower()
    assert len(data["sources"]) == 0


def test_end_to_end_ingest_and_ask(client):
    # Ingest file
    content = b"Full-time employees receive a one-time home office equipment stipend of $1200."
    res_ingest = client.post(
        "/documents/ingest",
        files={"file": ("stipend.txt", io.BytesIO(content), "text/plain")},
    )
    assert res_ingest.status_code == 200

    # Ask answerable question
    res_ask = client.post("/ask", json={"question": "What is the equipment stipend allowance?"})
    assert res_ask.status_code == 200
    data = res_ask.json()
    assert data["answerable"] is True
    assert "$1200" in data["answer"]
    assert len(data["sources"]) > 0
    assert data["sources"][0]["document"] == "stipend.txt"
    assert "$1200" in data["sources"][0]["excerpt"]


def test_get_document_content(client):
    # Ingest document
    content = b"Content for testing chunk viewing."
    client.post(
        "/documents/ingest",
        files={"file": ("view_test.txt", io.BytesIO(content), "text/plain")},
    )

    # Fetch document content
    res = client.get("/documents/view_test.txt")
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "view_test.txt"
    assert data["chunk_count"] > 0
    assert len(data["chunks"]) > 0
    assert "Content for testing chunk viewing" in data["chunks"][0]["text"]

    # Non-existent document
    res_404 = client.get("/documents/non_existent.txt")
    assert res_404.status_code == 404
