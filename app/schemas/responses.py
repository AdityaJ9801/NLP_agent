"""
Pydantic response models for all API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Sentiment ────────────────────────────────────────────────────────────────

class SentimentDistribution(BaseModel):
    count: int
    percentage: float


class SentimentResponse(BaseModel):
    total: int
    distribution: dict[str, SentimentDistribution]
    average_confidence: float
    per_row: list[dict[str, Any]] = Field(default_factory=list)


# ── NER ──────────────────────────────────────────────────────────────────────

class EntityTopValue(BaseModel):
    text: str
    count: int


class EntityTypeInfo(BaseModel):
    count: int
    top_values: list[EntityTopValue]


class NERResponse(BaseModel):
    total_entities: int
    entity_types: dict[str, EntityTypeInfo]


# ── Topics ───────────────────────────────────────────────────────────────────

class TopicInfo(BaseModel):
    id: int
    keywords: list[str]
    size: int


class TopicsResponse(BaseModel):
    num_topics: int
    topics: list[TopicInfo]
    outlier_count: int
    outlier_percentage: float
    total_documents: int
    message: str | None = None


# ── Classification ───────────────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    text: str
    label: str
    confidence: float
    error: str | None = None


class ClassifyResponse(BaseModel):
    total: int
    label_distribution: dict[str, Any]
    average_confidence: float
    results: list[ClassificationResult]


# ── Embeddings ───────────────────────────────────────────────────────────────

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    count: int
    storage: dict[str, Any] | None = None


# ── Summarization ────────────────────────────────────────────────────────────

class SummarizeResponse(BaseModel):
    summary: str
    chunks_processed: int
    method: str
    total_texts: int | None = None


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "nlp-text-agent"
    version: str = "1.0.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
    llm_provider: str = ""
    llm_model: str = ""
