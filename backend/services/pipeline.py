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
from services.scraper import scrape, scrape_rss_streaming, ScrapeParams
from services.embedder import embed_batch
from services.narrative_engine import batch_query_nearest, route_with_precomputed_nearest

# Chunk size for parallel Modal embed calls — avoids one massive payload and
# lets Modal schedule multiple GPU batches concurrently.
_EMBED_CHUNK_SIZE = 512

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
    Three-phase pipeline:

      Phase 1 — Parallel embed chunks
        Texts are split into _EMBED_CHUNK_SIZE chunks and embedded in
        concurrent Modal calls, reducing latency vs one giant payload.

      Phase 2 — Batch ChromaDB query
        batch_query_nearest() sends all embeddings to ChromaDB in chunks of
        256, acquiring _route_lock once per chunk instead of once per story.
        For 7 000 stories this is ~27 lock acquisitions vs 7 000.

      Phase 3 — Concurrent writes
        Each story's ChromaDB write (update or create) runs concurrently
        using the pre-computed nearest result from Phase 2.
    """
    if not stories:
        return

    texts = [f"{s.headline}\n\n{s.body}" for s in stories]
    logger.info(
        "[%s] %d fresh stories — embedding in %d parallel chunks…",
        label, len(stories), (len(texts) + _EMBED_CHUNK_SIZE - 1) // _EMBED_CHUNK_SIZE,
    )

    # ── Phase 1: parallel chunked embedding ───────────────────────────────────
    chunks = [texts[i : i + _EMBED_CHUNK_SIZE] for i in range(0, len(texts), _EMBED_CHUNK_SIZE)]
    try:
        chunk_results = await asyncio.gather(*[
            loop.run_in_executor(None, lambda c=chunk: embed_batch(c))
            for chunk in chunks
        ])
    except Exception as exc:
        logger.error("[%s] embed_batch failed: %s — aborting cycle", label, exc)
        pipeline_stats["errors"] += len(stories)
        return
    embeddings = [emb for chunk in chunk_results for emb in chunk]
    logger.info("[%s] embeddings done (%d vectors) — batch-querying ChromaDB…", label, len(embeddings))

    # ── Phase 2: batch ChromaDB query (one lock acquisition per 256 stories) ──
    try:
        nearest_per_story = await loop.run_in_executor(
            None, lambda: batch_query_nearest(embeddings)
        )
    except Exception as exc:
        logger.error("[%s] batch_query_nearest failed: %s — falling back to per-story routing", label, exc)
        nearest_per_story = [None] * len(stories)

    logger.info("[%s] query done — writing concurrently…", label)

    # ── Phase 3: concurrent writes (skip the ChromaDB query, already done) ────
    sem = asyncio.Semaphore(settings.pipeline_num_workers)

    async def _route_one(story, emb, nearest):
        async with sem:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda s=story, e=emb, n=nearest: route_with_precomputed_nearest(
                        s.headline, s.body, e, n
                    ),
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

    results = await asyncio.gather(*[
        _route_one(s, e, n) for s, e, n in zip(stories, embeddings, nearest_per_story)
    ])

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
    Streaming bulk ingest — pipelines scraping, embedding, and routing concurrently.

    Instead of:
        [scrape all 393 feeds ~20s] → [embed all ~15s] → [route all ~5s]

    This does:
        [scrape feeds...] ──→ queue ──→ [embed batch-1] ──→ [route batch-1]
                         ──→ queue ──→ [embed batch-2] ──→ [route batch-2]
                         ...

    A producer thread runs scrape_rss_streaming() (which uses as_completed
    internally), pushing per-feed story batches into an asyncio.Queue as each
    feed resolves.  The async consumer accumulates stories into STREAM_BATCH
    chunks and launches _process_stories() tasks concurrently — so batch-1 is
    being embedded/routed while batch-2 is still being scraped.
    """
    loop = asyncio.get_event_loop()
    lookback_minutes = settings.bulk_ingest_lookback_hours * 60

    params = ScrapeParams(
        lookback_minutes=lookback_minutes,
        max_per_source=settings.bulk_ingest_max_per_source,
        sources=["rss"],
    )

    n_feeds = len(params.rss_feeds)
    logger.info(
        "Bulk ingest starting (streaming) — lookback=%dh  max_per_feed=%d  feeds=%d",
        settings.bulk_ingest_lookback_hours,
        settings.bulk_ingest_max_per_source,
        n_feeds,
    )

    # Stories accumulate here; fire off a _process_stories task every STREAM_BATCH
    STREAM_BATCH = 1024

    # Queue carries per-feed story lists from the producer thread to the async consumer.
    # maxsize=8 provides backpressure so the thread doesn't race too far ahead.
    story_queue: asyncio.Queue = asyncio.Queue(maxsize=8)

    def _producer():
        """Run the blocking streaming generator; push per-feed batches into the queue."""
        try:
            for feed_batch in scrape_rss_streaming(params):
                asyncio.run_coroutine_threadsafe(
                    story_queue.put(feed_batch), loop
                ).result()
        finally:
            asyncio.run_coroutine_threadsafe(
                story_queue.put(None), loop  # sentinel
            ).result()

    # Launch producer in a thread so the async loop stays free
    producer_future = loop.run_in_executor(None, _producer)

    pending: list = []
    process_tasks: list[asyncio.Task] = []
    total_scraped = 0

    while True:
        feed_batch = await story_queue.get()
        if feed_batch is None:
            break  # producer finished

        pending.extend(feed_batch)
        total_scraped += len(feed_batch)

        # Kick off processing as soon as we have a full batch — don't await yet
        while len(pending) >= STREAM_BATCH:
            batch, pending = pending[:STREAM_BATCH], pending[STREAM_BATCH:]
            logger.info(
                "Bulk ingest: dispatching batch of %d (total scraped so far: %d)",
                len(batch), total_scraped,
            )
            process_tasks.append(
                asyncio.create_task(_process_stories(batch, loop, label="bulk"))
            )

    # Flush any remaining stories that didn't fill a full batch
    if pending:
        logger.info("Bulk ingest: flushing final %d stories", len(pending))
        process_tasks.append(
            asyncio.create_task(_process_stories(pending, loop, label="bulk"))
        )

    # Wait for the producer thread and all in-flight processing tasks
    await producer_future
    if process_tasks:
        await asyncio.gather(*process_tasks)

    logger.info(
        "Bulk ingest complete — %d stories across %d feeds",
        total_scraped, n_feeds,
    )


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
