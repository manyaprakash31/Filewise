"""Unit tests for document loader service."""

import pytest
from app.services.document_loader import DocumentLoader


def test_sanitize_filename():
    assert DocumentLoader.sanitize_filename("my file (1).pdf") == "my_file__1_.pdf"
    assert DocumentLoader.sanitize_filename("../../../etc/passwd") == "passwd"
    assert DocumentLoader.sanitize_filename("C:\\Windows\\System32\\calc.exe") == "calc.exe"
    assert DocumentLoader.sanitize_filename("") == "document"


def test_is_supported():
    assert DocumentLoader.is_supported("report.pdf") is True
    assert DocumentLoader.is_supported("notes.txt") is True
    assert DocumentLoader.is_supported("readme.md") is True
    assert DocumentLoader.is_supported("readme.markdown") is True
    assert DocumentLoader.is_supported("image.png") is False
    assert DocumentLoader.is_supported("archive.zip") is False


def test_load_txt_valid():
    content = b"This is a test plain text document with some facts.\nSecond line."
    pages = DocumentLoader.load_bytes("sample.txt", content)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "This is a test plain text document" in pages[0].text


def test_load_markdown_valid():
    content = b"# Title\n\n- Item 1\n- Item 2\n\nContent paragraph."
    pages = DocumentLoader.load_bytes("doc.md", content)
    assert len(pages) == 1
    assert "# Title" in pages[0].text
    assert "Item 1" in pages[0].text


def test_load_empty_file():
    with pytest.raises(ValueError, match="empty"):
        DocumentLoader.load_bytes("empty.txt", b"   ")


def test_load_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        DocumentLoader.load_bytes("data.csv", b"a,b,c\n1,2,3")


def test_load_pdf_malformed():
    with pytest.raises(ValueError, match="Failed to read PDF document|Malformed"):
        DocumentLoader.load_bytes("corrupted.pdf", b"This is definitely not a real PDF file header.")
