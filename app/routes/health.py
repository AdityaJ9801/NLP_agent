"""
GET /health — Service health check.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.schemas.responses import HealthResponse
from app.config import get_settings

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Return service health status including loaded models and uptime.
    """
    from app.pipelines import sentiment, ner, embeddings, topics

    models_loaded = {
        "sentiment": sentiment._pipeline_instance is not None,
        "spacy_ner": ner._nlp_instance is not None,
        "embeddings": embeddings._model_instance is not None,
        "bertopic": topics._model_instance is not None,
    }

    settings = get_settings()
    provider = str(getattr(settings, "LLM_PROVIDER", "unknown"))
    model = ""
    if provider == "azure_openai":
        model = getattr(settings, "AZURE_OPENAI_DEPLOYMENT_NAME", "")
    elif provider == "groq":
        model = getattr(settings, "GROQ_MODEL", "")
    elif provider == "openai":
        model = "gpt-4o"
    elif provider == "ollama":
        model = getattr(settings, "OLLAMA_MODEL", "")

    return HealthResponse(
        status="healthy",
        service="nlp-text-agent",
        version="1.0.0",
        models_loaded=models_loaded,
        uptime_seconds=round(time.time() - _start_time, 2),
        llm_provider=provider,
        llm_model=model,
    )
