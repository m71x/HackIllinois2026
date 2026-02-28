import time
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from services.narrative_engine import ingest_story
from services.scraper import scrape, ScrapeParams, cache_size
from services.story_buffer import buffer as story_buffer

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Manual single-story ingest (unchanged)
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    headline: str = Field(..., min_length=1, max_length=500)
    body: str = Field(default="", max_length=5000)
    source: str = Field(default="manual")


@router.post("")
def ingest(req: IngestRequest):
    return ingest_story(req.headline, req.body)


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
            results.append(ingest_story(story.headline, story.body))
        except Exception as e:
            errors.append(f"[{story.headline[:60]}]: {e}")
    return {
        "processed": len(results),
        "results": results,
        "errors": errors,
        "duration_seconds": round(time.time() - start, 2),
    }


# ---------------------------------------------------------------------------
# Scrape-and-ingest  ← the main controllable endpoint
# ---------------------------------------------------------------------------

class ScrapeRequest(BaseModel):
    """
    Controls a single scrape-and-ingest run.

    Time recency
    ------------
    lookback_minutes : int  (default 60)
        Only pull stories published within the last N minutes.
          30   → very fresh, low volume
          60   → last hour (default)
          360  → last 6 hours
          1440 → last 24 hours (NewsAPI free-tier max)

    Volume
    ------
    max_per_source : int  (default 50, max 100)
        Maximum stories to fetch from each source per run.

    Sources
    -------
    sources : list[str]  (default ["newsapi", "twitter"])
        Which sources to pull from. Any subset of:
          "newsapi"  — NewsAPI article summaries  (requires NEWSAPI_KEY)
          "twitter"  — Twitter/X recent tweets    (requires TWITTER_BEARER_TOKEN)

    Query overrides
    ---------------
    news_query : str | null
        Override the default financial/geopolitical keyword query sent to NewsAPI.
    twitter_query : str | null
        Override the default Twitter v2 search query.

    Other
    -----
    dry_run : bool  (default false)
        Fetch and deduplicate stories but do NOT ingest them into ChromaDB.
        Returns fetched stories for inspection. Useful for tuning params.

    buffer : bool  (default false)
        Fetch stories and hold them in the in-memory StoryBuffer instead of
        immediately embedding and routing. Call POST /api/pipeline/process
        when you are ready to commit them to ChromaDB.
    """
    lookback_minutes: int = Field(default=60, ge=1, le=10080)   # max 1 week
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
    buffer_mode: bool = False                      # true when stories were held in buffer
    buffer_size: Optional[int] = None             # total stories in buffer after this run
    stories_preview: Optional[list[dict]] = None  # only populated on dry_run


@router.post("/scrape", response_model=ScrapeRunResult)
async def scrape_and_ingest(req: ScrapeRequest):
    """
    Fetch fresh stories from configured sources and ingest them into ChromaDB.

    This is the primary way to populate the narrative database.
    All time-recency and volume controls are exposed as request parameters.
    """
    loop = asyncio.get_event_loop()

    # Build ScrapeParams, applying any query overrides
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

    # Count per-source before dedup by running source scrapers separately
    from services.scraper import scrape_newsapi, scrape_twitter, _cache
    per_source_raw: dict[str, int] = {}

    start = time.time()

    # Fetch all stories (runs in thread pool to avoid blocking event loop)
    stories = await loop.run_in_executor(None, lambda: scrape(params))

    # Tally per-source on raw fetch
    if "newsapi" in req.sources:
        raw_news = await loop.run_in_executor(None, lambda: scrape_newsapi(params))
        per_source_raw["newsapi"] = len(raw_news)
    if "twitter" in req.sources:
        raw_twitter = await loop.run_in_executor(None, lambda: scrape_twitter(params))
        per_source_raw["twitter"] = len(raw_twitter)

    total_raw = sum(per_source_raw.values())
    duplicates_skipped = total_raw - len(stories)

    # Dry run — return preview without ingesting
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
                for s in stories[:20]   # preview first 20
            ],
        )

    # Buffer mode — hold stories for later processing via POST /api/pipeline/process
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

    # Ingest each fresh story
    created = updated = errors = 0
    narratives_touched: dict[str, dict] = {}

    for story in stories:
        try:
            result = await loop.run_in_executor(
                None,
                lambda s=story: ingest_story(s.headline, s.body)
            )
            if result["action"] == "created":
                created += 1
            else:
                updated += 1

            # Track which narratives were touched (deduplicated by id)
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
