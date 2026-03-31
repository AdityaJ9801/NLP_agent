"""
NLP/Text Agent — Application configuration via pydantic-settings.
Reads from .env.free (default) or .env.paid depending on deployment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the NLP/Text Agent microservice."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env.paid"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ───────────────────────────────────────────────
    PORT: int = 8005
    APP_NAME: str = "NLP/Text Agent"
    APP_VERSION: str = "1.0.0"

    # ── LLM Provider ────────────────────────────────────────
    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "groq", "grok", "azure_openai"] = "azure_openai"

    # Ollama
    OLLAMA_BASE_URL: str = ""
    OLLAMA_MODEL: str = ""
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"

    # Groq
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    groq_api_key: str = ""
    xai_api_key: str = ""
    
    # OpenAI (optional)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ── NLP Models ──────────────────────────────────────────
    SPACY_MODEL: str = "en_core_web_sm"
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Processing ──────────────────────────────────────────
    NLP_BATCH_SIZE: int = 32
    MAX_TEXT_LENGTH: int = 10_000

    # ── Storage ─────────────────────────────────────────────
    HF_CACHE_DIR: str = "/app/hf_cache"
    REDIS_URL: str = "redis://redis:6379"
    VECTOR_STORE_TYPE: str = "none"  # none | pgvector
    PGVECTOR_URL: str = ""

    # ── Docker / EC2 ────────────────────────────────────────
    USE_DOCKER: bool = True
    USE_EC2: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
