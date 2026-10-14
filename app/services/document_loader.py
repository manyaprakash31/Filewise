"""Document loader service supporting PDF, TXT, and Markdown files."""

import io
import re
from pathlib import Path
from typing import List, Optional
import pypdf


class DocumentPage:
    """Represents text extracted from a single document or page."""

    def __init__(self, text: str, page_number: Optional[int] = None):
        self.text = text
        self.page_number = page_number


class DocumentLoader:
    """Extracts clean text and page numbers from supported file types."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent directory traversal and remove unsafe characters."""
        base = Path(filename).name
        # Keep alphanumeric, dots, underscores, dashes
        clean = re.sub(r"[^\w\-.]", "_", base)
        return clean or "document"

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """Check if the given filename has a supported extension."""
        return Path(filename).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def load_bytes(cls, filename: str, content: bytes) -> List[DocumentPage]:
        """Extract pages/sections from raw bytes given a filename.

        Raises:
            ValueError: If file is unsupported, empty, or unreadable.
        """
        if not content or len(content.strip()) == 0:
            raise ValueError("Uploaded file is empty.")

        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return cls._load_pdf(content)
        elif ext == ".txt":
            return cls._load_txt(content)
        elif ext in {".md", ".markdown"}:
            return cls._load_markdown(content)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Allowed: PDF, TXT, Markdown (.md).")

    @classmethod
    def _load_pdf(cls, content: bytes) -> List[DocumentPage]:
        pages: List[DocumentPage] = []
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            if len(reader.pages) == 0:
                raise ValueError("PDF file contains no pages.")

            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                cleaned = page_text.strip()
                if cleaned:
                    pages.append(DocumentPage(text=cleaned, page_number=idx))
        except pypdf.errors.PdfReadError as e:
            raise ValueError(f"Malformed or unreadable PDF: {str(e)}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to read PDF document: {str(e)}") from e

        if not pages:
            raise ValueError("PDF file contains no extractable text.")
        return pages

    @classmethod
    def _load_txt(cls, content: bytes) -> List[DocumentPage]:
        text = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError("Could not decode text file with standard encodings.")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text file is empty or contains only whitespace.")

        # TXT files have a single page (page_number=1)
        return [DocumentPage(text=cleaned, page_number=1)]

    @classmethod
    def _load_markdown(cls, content: bytes) -> List[DocumentPage]:
        text = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError("Could not decode Markdown file.")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Markdown file is empty or contains only whitespace.")

        # For retrieval, we keep the raw markdown text so headers & bullet semantics are retained
        return [DocumentPage(text=cleaned, page_number=1)]
