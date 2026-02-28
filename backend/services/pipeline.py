"""
Background ingestion pipeline.

Architecture
------------
Each poll cycle works in two phases:

  1. Batch-embed  — all fresh stories are embedded in a SINGLE Modal round-trip
                    via embed_batch(). This is the key throughput win over calling
                    embed_text() once per story.

  2. Concurrent route — each story is routed (LLM score + ChromaDB write) in its
                    own thread, limited to `pipeline_num_workers` concurrent workers.
                    The _route_lock inside narrative_engine serialises the brief
                    ChromaDB query+decision window; the slower Cerebras LLM calls
                    run fully in parallel.

Startup bulk ingest
-------------------
`bulk_ingest()` is called once at server start. It pulls up to
`bulk_ingest_lookback_hours` of RSS history (default 72 h) from all feeds and
processes the entire batch through the same two-phase pipeline. With Modal GPU
embeddings this typically finishes in a few minutes even for 1 000+ stories.
"""

import asyncio
import logging
import time
from core.config import settings
from services.scraper import scrape, ScrapeParams
from services.embedder import embed_batch
from services.narrative_engine import route_with_embedding

logger = logging.getLogger(__name__)

pipeline_stats = {
    "stories_ingested": 0,
    "narratives_created": 0,
    "narratives_updated": 0,
    "errors": 0,
    "started_at": None,
    "last_ingested_at": None,
    "last_poll_at": None,
}

_tasks: list[asyncio.Task] = []


# ---------------------------------------------------------------------------
# Core batch-processing function
# ---------------------------------------------------------------------------

async def _process_stories(stories, loop: asyncio.AbstractEventLoop, label: str = "poll"):
    """
    Embed `stories` in one batch call, then route each concurrently.
    Updates pipeline_stats and broadcasts SSE events.
    """
    if not stories:
        return

    logger.info("[%s] %d fresh stories — batch-embedding via Modal…", label, len(stories))

    # ── Phase 1: single Modal round-trip for all embeddings ──────────────────
    texts = [f"{s.headline}\n\n{s.body}" for s in stories]
    try:
        embeddings = await loop.run_in_executor(None, lambda: embed_batch(texts))
    except Exception as exc:
        logger.error("[%s] embed_batch failed: %s — aborting cycle", label, exc)
        pipeline_stats["errors"] += len(stories)
        return

    logger.info("[%s] embeddings done (%d vectors) — routing concurrently…", label, len(embeddings))

    # ── Phase 2: concurrent routing (LLM + ChromaDB) ─────────────────────────
    sem = asyncio.Semaphore(settings.pipeline_num_workers)

    async def _route_one(story, emb):
        async with sem:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda s=story, e=emb: route_with_embedding(s.headline, s.body, e),
                )
                pipeline_stats["stories_ingested"] += 1
                pipeline_stats["last_ingested_at"] = time.time()
                if result["action"] == "created":
                    pipeline_stats["narratives_created"] += 1
                else:
                    pipeline_stats["narratives_updated"] += 1

                try:
                    from api.routes.events import broadcast_event
                    broadcast_event({"type": "ingest", "timestamp": time.time(), "result": result})
                except Exception:
                    pass

                return result
            except Exception as exc:
                pipeline_stats["errors"] += 1
                logger.error("[%s] route failed [%s…]: %s", label, story.headline[:50], exc)
                return None

    results = await asyncio.gather(*[_route_one(s, e) for s, e in zip(stories, embeddings)])

    created = sum(1 for r in results if r and r["action"] == "created")
    updated = sum(1 for r in results if r and r["action"] == "updated")
    errors  = sum(1 for r in results if r is None)
    logger.info(
        "[%s] done — created=%d  updated=%d  errors=%d  total=%d",
        label, created, updated, errors, len(stories),
    )


# ---------------------------------------------------------------------------
# Regular polling loop
# ---------------------------------------------------------------------------

async def _poll_once(loop: asyncio.AbstractEventLoop):
    params = ScrapeParams(
        lookback_minutes=settings.pipeline_lookback_minutes,
        max_per_source=settings.pipeline_max_per_source,
        sources=settings.pipeline_sources,
    )
    try:
        stories = await loop.run_in_executor(None, lambda: scrape(params))
    except Exception as exc:
        logger.error("Pipeline scrape failed: %s", exc)
        return

    pipeline_stats["last_poll_at"] = time.time()
    await _process_stories(stories, loop, label="poll")


async def _pipeline_loop():
    loop = asyncio.get_event_loop()
    logger.info(
        "Pipeline started — interval=%ds  lookback=%dm  sources=%s  workers=%d",
        settings.poll_interval_seconds,
        settings.pipeline_lookback_minutes,
        settings.pipeline_sources,
        settings.pipeline_num_workers,
    )
    while True:
        await _poll_once(loop)
        await asyncio.sleep(settings.poll_interval_seconds)


# ---------------------------------------------------------------------------
# Startup bulk ingest — call once at server boot
# ---------------------------------------------------------------------------

async def bulk_ingest():
    """
    Pull the last `bulk_ingest_lookback_hours` of RSS history and ingest it all.

    Runs as a background task during server startup so it doesn't block the
    HTTP server from accepting requests. Progress is visible in server logs.
    """
    loop = asyncio.get_event_loop()
    lookback_minutes = settings.bulk_ingest_lookback_hours * 60

    logger.info(
        "Bulk ingest starting — lookback=%dh  max_per_feed=%d  feeds=%d",
        settings.bulk_ingest_lookback_hours,
        settings.bulk_ingest_max_per_source,
        # count only the rss feeds from a default ScrapeParams
        len(ScrapeParams().rss_feeds),
    )

    params = ScrapeParams(
        lookback_minutes=lookback_minutes,
        max_per_source=settings.bulk_ingest_max_per_source,
        sources=["rss"],   # RSS-only: no API keys needed, highest volume
    )

    try:
        stories = await loop.run_in_executor(None, lambda: scrape(params))
    except Exception as exc:
        logger.error("Bulk ingest scrape failed: %s", exc)
        return

    logger.info("Bulk ingest scraped %d fresh stories — beginning processing…", len(stories))
    await _process_stories(stories, loop, label="bulk")
    logger.info("Bulk ingest complete.")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def start_pipeline():
    pipeline_stats["started_at"] = time.time()
    _tasks.append(asyncio.create_task(_pipeline_loop()))


async def stop_pipeline():
    for task in _tasks:
        task.cancel()
    _tasks.clear()
