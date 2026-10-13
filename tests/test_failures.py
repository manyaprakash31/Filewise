"""Negative tests covering error conditions, validation failures, and edge cases."""

import io
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.llm_provider import LLMProvider
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore
from tests.test_retrieval import DummyEmbeddingService


class FailingLLMProvider(LLMProvider):
    """LLM provider that always raises an error simulating upstream API outage."""

    def generate_grounded_answer(self, question, context_chunks):
        raise RuntimeError("Upstream LLM API quota exceeded or service timed out.")


@pytest.fixture
def client_with_failing_llm():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_store = VectorStore(storage_dir=Path(tmpdir))
        test_embeddings = DummyEmbeddingService()
        test_retrieval = RetrievalService(test_store, test_embeddings)
        test_llm = FailingLLMProvider()
        test_qa = QAService(test_retrieval, test_llm)

        app = create_app()
        app.state.vector_store = test_store
        app.state.embedding_service = test_embeddings
        app.state.retrieval_service = test_retrieval
        app.state.llm_provider = test_llm
        app.state.qa_service = test_qa

        with TestClient(app) as test_client:
            yield test_client


def test_upload_invalid_extension(client_with_failing_llm):
    files = {"file": ("script.py", io.BytesIO(b"print('hello')"), "text/plain")}
    res = client_with_failing_llm.post("/documents/ingest", files=files)
    assert res.status_code == 400
    assert "Unsupported file extension" in res.json()["detail"]


def test_upload_empty_file(client_with_failing_llm):
    files = {"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")}
    res = client_with_failing_llm.post("/documents/ingest", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_invalid_ask_body(client_with_failing_llm):
    # Missing required 'question' parameter
    res = client_with_failing_llm.post("/ask", json={})
    assert res.status_code == 422  # Pydantic validation error


def test_llm_failure_handling(client_with_failing_llm):
    # Ingest document first so retrieval succeeds and invokes LLM
    content = b"The remote equipment stipend is $1200."
    res_ingest = client_with_failing_llm.post(
        "/documents/ingest",
        files={"file": ("stipend.txt", io.BytesIO(content), "text/plain")},
    )
    assert res_ingest.status_code == 200

    # Ask question that will trigger FailingLLMProvider
    res = client_with_failing_llm.post("/ask", json={"question": "What is the stipend allowance?"})
    assert res.status_code == 500
    assert "An error occurred while answering the question" in res.json()["detail"]


def test_batch_ingest_partial_failures(client_with_failing_llm):
    files = [
        ("files", ("valid.txt", io.BytesIO(b"Valid content for testing."), "text/plain")),
        ("files", ("bad.exe", io.BytesIO(b"Binary bytes"), "application/octet-stream")),
    ]
    res = client_with_failing_llm.post("/documents/ingest/batch", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_files"] == 2
    assert data["successful"] == 1
    assert data["failed"] == 1
    assert data["results"][0]["status"] == "indexed"
    assert data["results"][1]["status"] == "failed"
