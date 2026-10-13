"""Configuration settings for Filewise application."""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Filewise application settings loaded from environment or .env file."""

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # LLM Settings
    LLM_PROVIDER: Literal["mock", "gemini", "openai"] = "mock"
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = "gemini-2.5-flash"

    # Embeddings & Vector Store
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_STORE_DIR: str = "data/vector_store"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Retrieval
    RETRIEVAL_TOP_K: int = 4
    RELEVANCE_THRESHOLD: float = 0.35

    # Uploads
    MAX_UPLOAD_SIZE_MB: int = 15

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def vector_store_path(self) -> Path:
        """Absolute path to the vector store directory."""
        p = Path(self.VECTOR_STORE_DIR)
        if not p.is_absolute():
            return self.BASE_DIR / p
        return p

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()
