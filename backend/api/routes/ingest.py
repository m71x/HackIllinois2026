import time
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from services.narrative_engine import ingest_story
from core.state import pipeline_stats, broadcast_event

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Manual single-story ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    headline: str = Field(..., min_length=1, max_length=500)
    body: str = Field(default="", max_length=5000)
    source: str = Field(default="manual")


@router.post("")
def ingest(req: IngestRequest):
    try:
        result = ingest_story(req.headline, req.body)

        # Update pipeline stats
        pipeline_stats["stories_ingested"] += 1
        if result["action"] == "created":
            pipeline_stats["narratives_created"] += 1
        else:
            pipeline_stats["narratives_updated"] += 1

        # Broadcast to SSE subscribers
        broadcast_event({
            "type": "ingest",
            "result": result,
        })

        return result

    except Exception as e:
        pipeline_stats["errors"] += 1
        raise e


# ---------------------------------------------------------------------------
# Batch ingest
# ---------------------------------------------------------------------------

class BatchIngestRequest(BaseModel):
    stories: list[IngestRequest]
    max_stories: int = Field(default=50, le=100)


@router.post("/batch")
def ingest_batch(req: BatchIngestRequest):
    start = time.time()
    results, errors = [], []
    for story in req.stories[:req.max_stories]:
        try:
            result = ingest_story(story.headline, story.body)
            results.append(result)

            # Update pipeline stats
            pipeline_stats["stories_ingested"] += 1
            if result["action"] == "created":
                pipeline_stats["narratives_created"] += 1
            else:
                pipeline_stats["narratives_updated"] += 1

        except Exception as e:
            errors.append(f"[{story.headline[:60]}]: {e}")
            pipeline_stats["errors"] += 1

    return {
        "processed": len(results),
        "results": results,
        "errors": errors,
        "duration_seconds": round(time.time() - start, 2),
    }


# ---------------------------------------------------------------------------
# Scrape-and-ingest (optional - requires scraper service)
# ---------------------------------------------------------------------------

class ScrapeRequest(BaseModel):
    """
    Controls a single scrape-and-ingest run.
    """
    lookback_minutes: int = Field(default=60, ge=1, le=10080)
    max_per_source: int = Field(default=50, ge=1, le=100)
    sources: list[str] = Field(default=["newsapi", "twitter"])
    news_query: Optional[str] = None
    twitter_query: Optional[str] = None
    dry_run: bool = False
    buffer: bool = False


class ScrapeRunResult(BaseModel):
    fetched: int
    duplicates_skipped: int
    ingested: int
    narratives_created: int
    narratives_updated: int
    errors: int
    duration_seconds: float
    dedup_cache_size: int
    per_source: dict
    narratives_touched: list[dict]
    dry_run: bool
    buffer_mode: bool = False
    buffer_size: Optional[int] = None
    stories_preview: Optional[list[dict]] = None


@router.post("/scrape", response_model=ScrapeRunResult)
async def scrape_and_ingest(req: ScrapeRequest):
    """
    Fetch fresh stories from configured sources and ingest them into ChromaDB.
    Requires scraper service to be available.
    """
    try:
        from services.scraper import scrape, ScrapeParams, cache_size, scrape_newsapi, scrape_twitter
        from services.story_buffer import buffer as story_buffer
    except ImportError:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="Scraper service not available. Install newsapi-python and tweepy."
        )

    loop = asyncio.get_event_loop()

    params = ScrapeParams(
        lookback_minutes=req.lookback_minutes,
        max_per_source=req.max_per_source,
        sources=req.sources,
        dry_run=req.dry_run,
    )
    if req.news_query:
        params.news_query = req.news_query
    if req.twitter_query:
        params.twitter_query = req.twitter_query

    per_source_raw: dict[str, int] = {}
    start = time.time()

    stories = await loop.run_in_executor(None, lambda: scrape(params))

    if "newsapi" in req.sources:
        raw_news = await loop.run_in_executor(None, lambda: scrape_newsapi(params))
        per_source_raw["newsapi"] = len(raw_news)
    if "twitter" in req.sources:
        raw_twitter = await loop.run_in_executor(None, lambda: scrape_twitter(params))
        per_source_raw["twitter"] = len(raw_twitter)

    total_raw = sum(per_source_raw.values())
    duplicates_skipped = total_raw - len(stories)

    if req.dry_run:
        return ScrapeRunResult(
            fetched=len(stories),
            duplicates_skipped=duplicates_skipped,
            ingested=0,
            narratives_created=0,
            narratives_updated=0,
            errors=0,
            duration_seconds=round(time.time() - start, 2),
            dedup_cache_size=cache_size(),
            per_source=per_source_raw,
            narratives_touched=[],
            dry_run=True,
            stories_preview=[
                {"headline": s.headline[:200], "source": s.source, "body": s.body[:300]}
                for s in stories[:20]
            ],
        )

    if req.buffer:
        story_buffer.add_batch(stories)
        return ScrapeRunResult(
            fetched=len(stories),
            duplicates_skipped=duplicates_skipped,
            ingested=0,
            narratives_created=0,
            narratives_updated=0,
            errors=0,
            duration_seconds=round(time.time() - start, 2),
            dedup_cache_size=cache_size(),
            per_source=per_source_raw,
            narratives_touched=[],
            dry_run=False,
            buffer_mode=True,
            buffer_size=story_buffer.size(),
        )

    created = updated = errors = 0
    narratives_touched: dict[str, dict] = {}

    for story in stories:
        try:
            result = await loop.run_in_executor(
                None,
                lambda s=story: ingest_story(s.headline, s.body)
            )

            # Update pipeline stats
            pipeline_stats["stories_ingested"] += 1
            if result["action"] == "created":
                created += 1
                pipeline_stats["narratives_created"] += 1
            else:
                updated += 1
                pipeline_stats["narratives_updated"] += 1

            # Broadcast event
            broadcast_event({"type": "ingest", "result": result})

            nid = result["narrative_id"]
            narratives_touched[nid] = {
                "id": nid,
                "name": result["narrative_name"],
                "action": result["action"],
                "model_risk": result.get("model_risk"),
                "event_count": result.get("narrative_event_count"),
            }

        except Exception as e:
            errors += 1
            pipeline_stats["errors"] += 1
            logger.error(f"ingest_story failed for [{story.headline[:60]}]: {e}")

    return ScrapeRunResult(
        fetched=len(stories),
        duplicates_skipped=duplicates_skipped,
        ingested=created + updated,
        narratives_created=created,
        narratives_updated=updated,
        errors=errors,
        duration_seconds=round(time.time() - start, 2),
        dedup_cache_size=cache_size(),
        per_source=per_source_raw,
        narratives_touched=list(narratives_touched.values()),
        dry_run=False,
    )
