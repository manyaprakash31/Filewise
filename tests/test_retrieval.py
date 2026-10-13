"""Unit tests for semantic retrieval and vector store persistence."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from app.models.schemas import DocumentChunk
from app.services.embeddings import EmbeddingService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore


class DummyEmbeddingService:
    """Mock embedding service that maps terms to deterministic unit vectors."""

    def __init__(self):
        self.dimension = 4

    def embed_texts(self, texts):
        res = []
        for t in texts:
            t_lower = t.lower()
            if "kafka" in t_lower or "queue" in t_lower:
                vec = [1.0, 0.0, 0.0, 0.0]
            elif "vacation" in t_lower or "leave" in t_lower:
                vec = [0.0, 1.0, 0.0, 0.0]
            elif "stipend" in t_lower or "allowance" in t_lower:
                vec = [0.0, 0.0, 1.0, 0.0]
            else:
                vec = [0.0, 0.0, 0.0, 1.0]
            res.append(vec)
        return np.array(res, dtype=np.float32)

    def embed_query(self, query):
        return self.embed_texts([query])[0]


@pytest.fixture
def temp_vector_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(storage_dir=Path(tmpdir))
        yield store


def test_empty_knowledge_base_retrieval(temp_vector_store):
    dummy_embeddings = DummyEmbeddingService()
    retrieval = RetrievalService(temp_vector_store, dummy_embeddings)

    results = retrieval.retrieve("What is Kafka?")
    assert results == []


def test_semantic_retrieval_and_filtering(temp_vector_store):
    dummy_embeddings = DummyEmbeddingService()
    chunks = [
        DocumentChunk(
            chunk_id="doc1_000",
            document_name="architecture.txt",
            page_number=1,
            text="The ingestion pipeline uses Apache Kafka clusters across three zones.",
        ),
        DocumentChunk(
            chunk_id="doc2_000",
            document_name="policy.md",
            page_number=1,
            text="Employees receive an annual vacation accrual of 20 days.",
        ),
    ]
    embeddings = dummy_embeddings.embed_texts([c.text for c in chunks])
    temp_vector_store.add_chunks(chunks, embeddings)

    retrieval = RetrievalService(temp_vector_store, dummy_embeddings)

    # Search for Kafka (matches chunk 0 with score 1.0, chunk 1 with score 0.0)
    kafka_results = retrieval.retrieve("How does kafka message queue work?", top_k=2, relevance_threshold=0.5)
    assert len(kafka_results) == 1
    matched_chunk, score = kafka_results[0]
    assert matched_chunk.document_name == "architecture.txt"
    assert "Kafka" in matched_chunk.text
    assert score >= 0.99


def test_vector_store_persistence(temp_vector_store):
    dummy_embeddings = DummyEmbeddingService()
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_name="doc.txt",
            page_number=1,
            text="Test persistence text.",
        )
    ]
    temp_vector_store.add_chunks(chunks, dummy_embeddings.embed_texts(["Test persistence text."]))

    # Reload from the same directory
    reloaded_store = VectorStore(storage_dir=temp_vector_store.storage_dir)
    assert reloaded_store.total_chunks() == 1
    docs = reloaded_store.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "doc.txt"
    assert docs[0].chunk_count == 1
