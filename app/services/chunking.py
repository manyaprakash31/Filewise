"""Text chunking service with configurable chunk size and overlap."""

import re
from typing import List, Optional
from app.models.schemas import DocumentChunk
from app.services.document_loader import DocumentPage


class TextChunker:
    """Splits document text into overlapping chunks while preserving sentence/paragraph boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document_name: str,
        pages: List[DocumentPage],
    ) -> List[DocumentChunk]:
        """Split a list of DocumentPages into DocumentChunks with unique IDs and metadata."""
        chunks: List[DocumentChunk] = []
        global_idx = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            page_chunks = self._split_text(page_text)
            for chunk_str in page_chunks:
                cid = f"{document_name}_{global_idx:03d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        document_name=document_name,
                        page_number=page.page_number,
                        text=chunk_str,
                    )
                )
                global_idx += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into pieces respecting chunk_size and chunk_overlap using recursive paragraph/sentence splits."""
        if len(text) <= self.chunk_size:
            return [text]

        # Break text into paragraphs / double newlines first
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        pieces: List[str] = []
        for p in paragraphs:
            if len(p) <= self.chunk_size:
                pieces.append(p)
            else:
                # Break large paragraph by sentences
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
                for s in sentences:
                    if len(s) <= self.chunk_size:
                        pieces.append(s)
                    else:
                        # Fallback: slice by words
                        words = s.split()
                        curr_words: List[str] = []
                        curr_len = 0
                        for w in words:
                            if curr_len + len(w) + 1 > self.chunk_size:
                                if curr_words:
                                    pieces.append(" ".join(curr_words))
                                curr_words = [w]
                                curr_len = len(w)
                            else:
                                curr_words.append(w)
                                curr_len += len(w) + 1
                        if curr_words:
                            pieces.append(" ".join(curr_words))

        # Now group pieces into overlapping chunks
        chunks: List[str] = []
        curr_chunk_pieces: List[str] = []
        curr_len = 0

        for piece in pieces:
            added_len = len(piece) + (1 if curr_chunk_pieces else 0)
            if curr_len + added_len > self.chunk_size and curr_chunk_pieces:
                # Commit current chunk
                combined = " ".join(curr_chunk_pieces).strip()
                chunks.append(combined)

                # Backtrack to implement overlap
                overlap_pieces: List[str] = []
                overlap_len = 0
                for prev in reversed(curr_chunk_pieces):
                    if overlap_len + len(prev) + 1 <= self.chunk_overlap:
                        overlap_pieces.insert(0, prev)
                        overlap_len += len(prev) + 1
                    else:
                        break

                curr_chunk_pieces = overlap_pieces + [piece]
                curr_len = sum(len(p) for p in curr_chunk_pieces) + (len(curr_chunk_pieces) - 1)
            else:
                curr_chunk_pieces.append(piece)
                curr_len += added_len

        if curr_chunk_pieces:
            combined = " ".join(curr_chunk_pieces).strip()
            if not chunks or combined != chunks[-1]:
                chunks.append(combined)

        return chunks
