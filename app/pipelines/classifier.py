"""
Zero-shot text classification pipeline.

Delegates to the configured LLM provider (Ollama / Groq) for inference.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def classify_batch(
    texts: list[str],
    labels: list[str],
) -> list[dict[str, Any]]:
    """
    Classify each text into one of *labels* using zero-shot LLM inference.

    Returns a list of dicts: ``{"text": str, "label": str, "confidence": float}``
    """
    from app.llm.provider import get_llm_provider

    provider = get_llm_provider()
    results = []

    for text in texts:
        try:
            result = await provider.classify(text, labels)
            results.append(result)
        except Exception as e:
            logger.error("Classification failed for text: %s", e)
            results.append({
                "text": text[:100],
                "label": "unknown",
                "confidence": 0.0,
                "error": str(e),
            })

    return results


def aggregate_classifications(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate classification results into a label-level distribution.

    Returns::

        {
            "total": int,
            "label_distribution": {
                "label_a": {"count": int, "percentage": float},
                ...
            },
            "average_confidence": float,
        }
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "label_distribution": {}, "average_confidence": 0.0}

    label_counts: dict[str, int] = {}
    total_confidence = 0.0

    for r in results:
        label = r.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
        total_confidence += r.get("confidence", 0.0)

    label_distribution = {
        label: {
            "count": count,
            "percentage": round((count / total) * 100, 2),
        }
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])
    }

    return {
        "total": total,
        "label_distribution": label_distribution,
        "average_confidence": round(total_confidence / total, 4),
    }
