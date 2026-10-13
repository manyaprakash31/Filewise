"""Embeddings service providing local embeddings via SentenceTransformers with lazy loading."""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages local embedding generation using sentence-transformers."""

    _instance = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            logger.info("Loading sentence-transformers model: %s", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error("Failed to load embedding model %s: %s", self.model_name, e)
                raise RuntimeError(
                    f"Could not load local embedding model '{self.model_name}'. "
                    f"Ensure PyTorch and sentence-transformers are properly installed. Error: {e}"
                ) from e
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized float32 embeddings for a list of texts."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._get_model()
        # normalize_embeddings=True allows cosine similarity via inner product
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized float32 embedding for a single query."""
        res = self.embed_texts([query])
        return res[0]

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings for the configured model."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
