"""
Background ingestion pipeline.

Runs on a fixed schedule (poll_interval_seconds), fetching fresh stories
from all configured sources and routing them through narrative_engine.

The same dedup cache used by the manual /api/ingest/scrape endpoint is
shared here, so manual and automatic scrapes never double-ingest.
"""

import asyncio
import logging
import time
from core.config import settings
from services.scraper import scrape, ScrapeParams
from services.narrative_engine import ingest_story

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


async def _poll_once(loop: asyncio.AbstractEventLoop):
    """Run one scrape-and-ingest cycle."""
    params = ScrapeParams(
        lookback_minutes=settings.pipeline_lookback_minutes,
        max_per_source=settings.pipeline_max_per_source,
        sources=settings.pipeline_sources,
    )

    try:
        stories = await loop.run_in_executor(None, lambda: scrape(params))
    except Exception as e:
        logger.error(f"Pipeline scrape failed: {e}")
        return

    pipeline_stats["last_poll_at"] = time.time()

    for story in stories:
        try:
            result = await loop.run_in_executor(
                None,
                lambda s=story: ingest_story(s.headline, s.body)
            )
            pipeline_stats["stories_ingested"] += 1
            pipeline_stats["last_ingested_at"] = time.time()

            if result["action"] == "created":
                pipeline_stats["narratives_created"] += 1
            else:
                pipeline_stats["narratives_updated"] += 1

            # Broadcast to SSE clients if available
            try:
                from api.routes.events import broadcast_event
                broadcast_event({
                    "type": "ingest",
                    "timestamp": time.time(),
                    "result": result,
                })
            except Exception:
                pass

        except Exception as e:
            pipeline_stats["errors"] += 1
            logger.error(f"Pipeline ingest failed [{story.headline[:60]}]: {e}")


async def _pipeline_loop():
    loop = asyncio.get_event_loop()
    logger.info(
        f"Pipeline started — polling every {settings.poll_interval_seconds}s, "
        f"lookback={settings.pipeline_lookback_minutes}m, "
        f"sources={settings.pipeline_sources}"
    )
    while True:
        await _poll_once(loop)
        await asyncio.sleep(settings.poll_interval_seconds)


async def start_pipeline():
    pipeline_stats["started_at"] = time.time()
    _tasks.append(asyncio.create_task(_pipeline_loop()))


async def stop_pipeline():
    for task in _tasks:
        task.cancel()
    _tasks.clear()
