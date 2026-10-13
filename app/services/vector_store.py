"""Persistent FAISS vector store with JSON metadata serialization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
from app.models.schemas import DocumentChunk, IndexedDocument

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent vector store combining FAISS IndexFlatIP with JSON metadata."""

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.storage_dir / self.INDEX_FILENAME
        self.metadata_file = self.storage_dir / self.METADATA_FILENAME

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self._load()

    def _load(self) -> None:
        """Load index and chunk metadata from disk if available."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    raw_chunks = json.load(f)
                    self.chunks = [DocumentChunk(**item) for item in raw_chunks]
                logger.info(
                    "Loaded persistent vector store from %s with %d chunks.",
                    self.storage_dir,
                    len(self.chunks),
                )
            except Exception as e:
                logger.error("Failed to load vector store from %s: %s", self.storage_dir, e)
                self.index = None
                self.chunks = []

    def _save(self) -> None:
        """Persist index and chunk metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump([chunk.model_dump() for chunk in self.chunks], f, indent=2)
            logger.info("Persisted vector store with %d chunks to %s", len(self.chunks), self.storage_dir)

    def is_document_indexed(self, filename: str) -> bool:
        """Check if any chunks from this document are already present in the store."""
        return any(c.document_name == filename for c in self.chunks)

    def add_chunks(self, new_chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add new document chunks and corresponding embeddings into the store and save."""
        if not new_chunks or len(embeddings) == 0:
            return

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        elif self.index.d != dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.index.d}, got {dim}")

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self._save()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        relevance_threshold: float = 0.35,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search the store for chunks matching the query embedding above the similarity threshold."""
        if self.index is None or self.index.ntotal == 0 or not self.chunks:
            return []

        # Ensure query embedding shape (1, d)
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            similarity = float(score)
            if similarity >= relevance_threshold:
                results.append((self.chunks[idx], similarity))

        return results

    def list_documents(self) -> List[IndexedDocument]:
        """Summarize all indexed documents and their chunk counts."""
        doc_map: Dict[str, Dict] = {}
        for c in self.chunks:
            if c.document_name not in doc_map:
                doc_map[c.document_name] = {"chunk_count": 0, "pages": set()}
            doc_map[c.document_name]["chunk_count"] += 1
            if c.page_number is not None:
                doc_map[c.document_name]["pages"].add(c.page_number)

        summary: List[IndexedDocument] = []
        for name, data in doc_map.items():
            page_count = len(data["pages"]) if data["pages"] else None
            summary.append(
                IndexedDocument(
                    filename=name,
                    chunk_count=data["chunk_count"],
                    page_count=page_count,
                )
            )
        return sorted(summary, key=lambda d: d.filename)

    def get_document_chunks(self, filename: str) -> List[DocumentChunk]:
        """Return all indexed chunks belonging to the specified document filename."""
        return [c for c in self.chunks if c.document_name == filename]

    def total_chunks(self) -> int:
        """Total number of indexed chunks."""
        return len(self.chunks)

    def clear(self) -> None:
        """Clear all stored embeddings and metadata."""
        self.index = None
        self.chunks = []
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
