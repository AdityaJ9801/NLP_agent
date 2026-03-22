"""
POST /topics — Topic modeling via BERTopic on a text column.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from app.pipelines import topics as topics_pipeline
from app.schemas.requests import TextListRequest
from app.schemas.responses import TopicsResponse
from app.utils.text_cleaner import clean_texts

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/topics", response_model=TopicsResponse)
async def model_topics(request: TextListRequest):
    """
    Fit a BERTopic model on the provided texts.

    Returns topic keywords, sizes, and outlier percentage.
    Requires at least 10 documents for meaningful results.
    """
    logger.info("Topic modeling requested for %d texts", len(request.texts))

    cleaned = clean_texts(request.texts)

    # BERTopic is CPU-heavy — run in executor
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, topics_pipeline.fit_topics, cleaned)

    return TopicsResponse(**result)
