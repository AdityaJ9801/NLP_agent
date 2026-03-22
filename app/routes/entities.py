"""
POST /entities — Named Entity Recognition (NER) on a text column.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.pipelines import ner as ner_pipeline
from app.schemas.requests import TextListRequest
from app.schemas.responses import NERResponse
from app.utils.batch_processor import process_text_column
from app.utils.text_cleaner import clean_texts

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/entities", response_model=NERResponse)
async def extract_entities(request: TextListRequest):
    """
    Extract named entities from a list of texts using spaCy.

    Returns entity types with top-10 values per type and total counts.
    """
    logger.info("NER requested for %d texts", len(request.texts))

    cleaned = clean_texts(request.texts)
    all_entities = await process_text_column(cleaned, ner_pipeline.extract_batch)
    aggregated = ner_pipeline.aggregate(all_entities)

    return NERResponse(**aggregated)
