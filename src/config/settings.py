"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the recommendation engine."""

    model_config = {"env_prefix": "REC_", "env_file": ".env", "extra": "ignore"}

    # Storage
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_DIR: str = "./data/chroma"
    DATA_DIR: str = "./data/store"

    # Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    SVD_FACTORS: int = 50
    KNN_NEIGHBORS: int = 20

    # Serving
    TOP_K_DEFAULT: int = 10
    MIN_SIMILARITY: float = 0.01
    CACHE_TTL: int = 300  # seconds
    CACHE_MAX_SIZE: int = 10_000

    # Real-time
    BATCH_SIZE: int = 100
    TRENDING_WINDOW_HOURS: int = 24

    # A/B testing
    AB_MIN_SAMPLE_SIZE: int = 100
    AB_SIGNIFICANCE_LEVEL: float = 0.05

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
