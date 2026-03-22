"""
POST /classify — Zero-shot text classification.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.pipelines import classifier
from app.schemas.requests import ClassifyRequest
from app.schemas.responses import ClassifyResponse
from app.utils.text_cleaner import clean_texts

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/classify", response_model=ClassifyResponse)
async def classify_texts(request: ClassifyRequest):
    """
    Classify each text into one of the provided candidate labels
    using zero-shot LLM inference.
    """
    logger.info(
        "Classification requested for %d texts with labels: %s",
        len(request.texts),
        request.labels,
    )

    cleaned = clean_texts(request.texts)
    results = await classifier.classify_batch(cleaned, request.labels)
    aggregated = classifier.aggregate_classifications(results)

    return ClassifyResponse(
        total=aggregated["total"],
        label_distribution=aggregated["label_distribution"],
        average_confidence=aggregated["average_confidence"],
        results=results,
    )
