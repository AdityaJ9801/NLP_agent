"""
Topic Modeling pipeline — BERTopic.

BERTopic is initialised once with nr_topics=10, min_topic_size=5.
Uses SentenceTransformer embeddings internally.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Singleton ────────────────────────────────────────────────────────────────

_model_instance = None


def get_model():
    """Return the cached BERTopic model (lazy-loaded)."""
    global _model_instance
    if _model_instance is None:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        logger.info("Initializing BERTopic with embedding model: %s", settings.EMBEDDING_MODEL)

        embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            cache_folder=settings.HF_CACHE_DIR,
        )
        _model_instance = BERTopic(
            embedding_model=embedding_model,
            nr_topics=10,
            min_topic_size=5,
            verbose=False,
        )
        logger.info("BERTopic model initialized successfully.")
    return _model_instance


# ── Fit & transform ─────────────────────────────────────────────────────────

def fit_topics(texts: list[str]) -> dict[str, Any]:
    """
    Fit BERTopic on *texts* and return structured topic results.

    Returns::

        {
            "num_topics": int,
            "topics": [
                {
                    "id": int,
                    "keywords": list[str],
                    "size": int,
                },
                ...
            ],
            "outlier_count": int,
            "outlier_percentage": float,
            "total_documents": int,
        }
    """
    model = get_model()

    if len(texts) < 10:
        return {
            "num_topics": 0,
            "topics": [],
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "total_documents": len(texts),
            "message": "Insufficient documents for topic modeling (need ≥ 10).",
        }

    topics, _probs = model.fit_transform(texts)

    # Count outliers (topic == -1)
    outlier_count = sum(1 for t in topics if t == -1)
    total = len(topics)

    # Get topic info
    topic_info = model.get_topic_info()
    topic_list = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue  # skip outlier cluster
        keywords = [word for word, _ in model.get_topic(topic_id)][:10]
        topic_list.append({
            "id": int(topic_id),
            "keywords": keywords,
            "size": int(row["Count"]),
        })

    return {
        "num_topics": len(topic_list),
        "topics": topic_list,
        "outlier_count": outlier_count,
        "outlier_percentage": round((outlier_count / total) * 100, 2) if total else 0.0,
        "total_documents": total,
    }
