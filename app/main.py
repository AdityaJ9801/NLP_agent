"""
NLP/Text Agent — FastAPI Application Entrypoint.

Initialises NLP pipeline singletons at startup via @app.on_event("startup").
Mounts all API routers for the 7 endpoints.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nlp-text-agent")

# Suppress noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── App ──────────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Unstructured Text & Semantic Specialist — processes text columns "
        "and documents through batched NLP pipelines (spaCy, Transformers, "
        "BERTopic, SentenceTransformers)."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup — lazy-load NLP singletons ───────────────────────────────────────


@app.on_event("startup")
async def startup_load_models():
    """
    Pre-load all NLP pipeline singletons at server startup.
    Models are loaded once and shared across all requests.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("NLP/Text Agent starting up…")
    logger.info("LLM Provider : %s", settings.LLM_PROVIDER)
    logger.info("spaCy Model  : %s", settings.SPACY_MODEL)
    logger.info("Sentiment    : %s", settings.SENTIMENT_MODEL)
    logger.info("Embeddings   : %s", settings.EMBEDDING_MODEL)
    logger.info("Batch Size   : %d", settings.NLP_BATCH_SIZE)
    logger.info("=" * 60)

    # Load in sequence (each model logs its own progress)
    try:
        from app.pipelines.sentiment import get_pipeline
        get_pipeline()
    except Exception as e:
        logger.warning("Sentiment model load deferred: %s", e)

    try:
        from app.pipelines.ner import get_nlp
        get_nlp()
    except Exception as e:
        logger.warning("spaCy model load deferred: %s", e)

    try:
        from app.pipelines.embeddings import get_model as get_embed_model
        get_embed_model()
    except Exception as e:
        logger.warning("Embedding model load deferred: %s", e)

    # BERTopic is loaded on first /topics call (heavier init)
    # LLM provider is loaded on first /classify or /summarize call

    elapsed = round(time.time() - start, 2)
    logger.info("Startup complete in %.2f seconds.", elapsed)


# ── Routers ──────────────────────────────────────────────────────────────────

from app.routes.sentiment import router as sentiment_router
from app.routes.entities import router as entities_router
from app.routes.topics import router as topics_router
from app.routes.classify import router as classify_router
from app.routes.embed import router as embed_router
from app.routes.summarize import router as summarize_router
from app.routes.health import router as health_router

app.include_router(sentiment_router, tags=["Sentiment"])
app.include_router(entities_router, tags=["NER"])
app.include_router(topics_router, tags=["Topics"])
app.include_router(classify_router, tags=["Classification"])
app.include_router(embed_router, tags=["Embeddings"])
app.include_router(summarize_router, tags=["Summarization"])
app.include_router(health_router, tags=["Health"])

# ── Root ─────────────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    return {
        "service": "NLP/Text Agent",
        "version": settings.APP_VERSION,
        "port": settings.PORT,
        "docs": "/docs",
    }


# ── Orchestrator integration ──────────────────────────────────────────────────


@app.post("/run", tags=["Orchestrator"])
async def run_task(payload: dict):
    """
    Orchestrator pipeline integration endpoint.

    Extracts text columns from upstream context (_context) and runs
    sentiment analysis + NER. Returns a combined NLP insights dict
    that the Report Agent can consume.

    Expected _context keys (from Context or SQL agent):
      - data_preview: list[dict]  — rows with text columns
      - columns: list             — column metadata (used to find text cols)
    """
    from app.pipelines import sentiment as sentiment_pipeline, ner as ner_pipeline
    from app.utils.batch_processor import process_text_column
    from app.utils.text_cleaner import clean_texts

    context = payload.get("_context", {})
    task_description = payload.get("task_description") or payload.get("query") or ""

    data_rows: list[dict] = []
    column_meta: list = []

    # Locate rows — prefer SQL agent output, fall back to Context Agent sample_values
    for dep_data in context.values():
        if not isinstance(dep_data, dict):
            continue
        if "data_preview" in dep_data and dep_data["data_preview"]:
            data_rows = dep_data["data_preview"]
            column_meta = dep_data.get("columns", [])
            break
        if "source_id" in dep_data and "columns" in dep_data and not data_rows:
            # Reconstruct rows from sample_values
            cols = dep_data["columns"]
            if cols:
                max_samples = max(len(c.get("sample_values", [])) for c in cols)
                for i in range(max_samples):
                    row = {}
                    for col in cols:
                        samples = col.get("sample_values", [])
                        row[col["name"]] = samples[i] if i < len(samples) else None
                    data_rows.append(row)
                column_meta = cols

    if not data_rows:
        return {
            "nlp_type": "no_data",
            "message": "No text data found in _context",
            "sentiment": None,
            "entities": None,
        }

    # Identify text columns (object/string dtype, not identifiers)
    _SKIP_SEMANTICS = {"identifier", "datetime", "date", "url", "email"}
    text_col_names: list[str] = []
    for col in column_meta:
        if isinstance(col, dict):
            dtype = col.get("dtype", "")
            semantic = col.get("semantic_type", "")
            name = col.get("name", "")
            if dtype in ("object", "string") and semantic not in _SKIP_SEMANTICS:
                text_col_names.append(name)
        elif isinstance(col, str):
            # Infer from first row value
            val = data_rows[0].get(col, "") if data_rows else ""
            if isinstance(val, str) and len(val) > 3:
                text_col_names.append(col)

    # Fallback: pick any string column with avg length > 5
    if not text_col_names and data_rows:
        for key, val in data_rows[0].items():
            if isinstance(val, str) and len(val) > 5:
                text_col_names.append(key)

    if not text_col_names:
        return {
            "nlp_type": "no_text_columns",
            "message": "No suitable text columns found for NLP analysis",
            "sentiment": None,
            "entities": None,
        }

    # Collect texts from the first suitable text column
    primary_col = text_col_names[0]
    texts = [str(r[primary_col]) for r in data_rows if r.get(primary_col)]

    if not texts:
        return {
            "nlp_type": "empty_texts",
            "message": f"Column '{primary_col}' has no non-null values",
            "sentiment": None,
            "entities": None,
        }

    cleaned = clean_texts(texts)

    # Run sentiment + NER in parallel
    import asyncio as _asyncio
    sentiment_rows, entity_rows = await _asyncio.gather(
        process_text_column(cleaned, sentiment_pipeline.analyze_batch),
        process_text_column(cleaned, ner_pipeline.extract_batch),
    )

    sentiment_agg = sentiment_pipeline.aggregate(sentiment_rows)
    entity_agg = ner_pipeline.aggregate(entity_rows)

    return {
        "nlp_type": "sentiment_ner",
        "text_column": primary_col,
        "total_texts": len(texts),
        "sentiment": sentiment_agg,
        "entities": entity_agg,
        "task": task_description,
    }
