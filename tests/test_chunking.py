"""Unit tests for text chunker service."""

import pytest
from app.services.chunking import TextChunker
from app.services.document_loader import DocumentPage


def test_chunker_validation():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        TextChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_short_text():
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    pages = [DocumentPage(text="Short document that fits easily inside one chunk.", page_number=1)]
    chunks = chunker.chunk_document("test.txt", pages)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test.txt_000"
    assert chunks[0].document_name == "test.txt"
    assert chunks[0].page_number == 1
    assert "Short document" in chunks[0].text


def test_chunker_splits_and_overlap():
    chunker = TextChunker(chunk_size=150, chunk_overlap=40)
    long_text = (
        "Meridian Technologies provides a monthly internet connectivity subsidy of $75 to offset broadband costs. "
        "To qualify for reimbursement, residential download speeds must meet or exceed 100 Mbps. "
        "Expense reimbursement requests must be submitted through the Expensify portal by the 5th business day."
    )
    pages = [DocumentPage(text=long_text, page_number=1)]
    chunks = chunker.chunk_document("policy.md", pages)

    assert len(chunks) > 1
    for c in chunks:
        assert c.document_name == "policy.md"
        assert c.page_number == 1
        assert len(c.text) > 0


def test_chunker_multiple_pages_preserves_page_numbers():
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    pages = [
        DocumentPage(text="Chapter 1 content on page one.", page_number=1),
        DocumentPage(text="Chapter 2 content on page two.", page_number=2),
    ]
    chunks = chunker.chunk_document("handbook.pdf", pages)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].chunk_id == "handbook.pdf_000"
    assert chunks[1].chunk_id == "handbook.pdf_001"
