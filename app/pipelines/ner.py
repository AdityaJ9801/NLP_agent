"""
Named Entity Recognition pipeline — spaCy.

Uses en_core_web_sm (free) or en_core_web_lg (paid).
Loaded once at startup as a singleton via ``get_nlp()``.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Singleton ────────────────────────────────────────────────────────────────

_nlp_instance = None


def get_nlp():
    """Return the cached spaCy Language model (lazy-loaded)."""
    global _nlp_instance
    if _nlp_instance is None:
        import spacy

        settings = get_settings()
        model_name = settings.SPACY_MODEL
        logger.info("Loading spaCy model: %s", model_name)
        _nlp_instance = spacy.load(model_name, disable=["parser", "lemmatizer"])
        logger.info("spaCy model loaded successfully.")
    return _nlp_instance


# ── Batch extraction ────────────────────────────────────────────────────────

def extract_batch(texts: list[str]) -> list[list[dict[str, str]]]:
    """
    Extract named entities from a batch of texts.

    Returns a list (per text) of entity dicts:
    ``[{"text": "Apple", "label": "ORG", "start": 0, "end": 5}, ...]``
    """
    nlp = get_nlp()
    results = []
    for doc in nlp.pipe(texts, batch_size=len(texts)):
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        results.append(entities)
    return results


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(all_entities: list[list[dict[str, str]]]) -> dict[str, Any]:
    """
    Aggregate entities across all texts into a type-level summary.

    Returns::

        {
            "total_entities": int,
            "entity_types": {
                "ORG": {
                    "count": int,
                    "top_values": [{"text": str, "count": int}, ...],  # top 10
                },
                ...
            }
        }
    """
    type_counters: dict[str, Counter] = defaultdict(Counter)
    total = 0

    for text_entities in all_entities:
        for ent in text_entities:
            type_counters[ent["label"]][ent["text"]] += 1
            total += 1

    entity_types = {}
    for ent_type, counter in sorted(type_counters.items()):
        top_values = [
            {"text": text, "count": count}
            for text, count in counter.most_common(10)
        ]
        entity_types[ent_type] = {
            "count": sum(counter.values()),
            "top_values": top_values,
        }

    return {
        "total_entities": total,
        "entity_types": entity_types,
    }
