"""Dedicated semantic retrieval service."""

import logging
from typing import List, Tuple
from app.models.schemas import DocumentChunk
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """Orchestrates query embedding, semantic similarity search, and threshold filtering."""

    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        relevance_threshold: float = 0.35,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve the most relevant document chunks for a given query above the threshold.

        Returns:
            List of (DocumentChunk, similarity_score) tuples sorted by descending similarity.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        # If vector store is empty, avoid unnecessary embedding compute
        if self.vector_store.total_chunks() == 0:
            logger.info("Knowledge base is empty. Returning empty retrieval result.")
            return []

        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(cleaned_query)

        # Query vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )

        logger.info(
            "Query '%s' returned %d chunks (top_k=%d, threshold=%.2f)",
            cleaned_query,
            len(results),
            top_k,
            relevance_threshold,
        )
        return results
