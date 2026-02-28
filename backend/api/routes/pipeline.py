"""
Pipeline Routes
===============
On-demand controls for the staged ingestion flow.

Staged flow
-----------
1.  POST /api/ingest/scrape  {"buffer": true}
        Scrapes stories → holds them in StoryBuffer. Nothing hits Modal or ChromaDB yet.

2.  GET  /api/pipeline/buffer
        Inspect what is waiting. Check before committing.

3.  POST /api/pipeline/process
        Drain the buffer → batch-embed (one Modal call) → route each story into ChromaDB.

4.  DELETE /api/pipeline/buffer
        Discard buffered stories without processing them.

5.  GET  /api/pipeline/stats
        Session-wide counters for the background auto-pipeline (if running).
"""

import time
import asyncio
import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from services.story_buffer import buffer as story_buffer
from services.embedder import embed_batch
from services.narrative_engine import route_with_embedding
from services.pipeline import pipeline_stats
from db import vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# GET /api/pipeline/buffer  — inspect what is waiting
# ---------------------------------------------------------------------------

@router.get("/buffer")
def get_buffer_stats():
    """
    Return a summary of stories currently waiting in the buffer.

    Fields
    ------
    count       : total stories waiting
    oldest_at   : unix timestamp of oldest story (null if empty)
    newest_at   : unix timestamp of newest story (null if empty)
    preview     : first 10 stories as {headline, source}
    """
    stats = story_buffer.stats(preview_limit=10)
    return {
        "count": stats.count,
        "oldest_at": stats.oldest_at,
        "newest_at": stats.newest_at,
        "preview": stats.preview,
    }


# ---------------------------------------------------------------------------
# POST /api/pipeline/process  — batch-embed and route everything in the buffer
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    max_stories: int = Field(
        default=500, ge=1, le=2000,
        description="Cap on how many buffered stories to process in one call.",
    )


@router.post("/process")
async def process_buffer(req: ProcessRequest):
    """
    Drain the story buffer, batch-embed via Modal (one network round-trip),
    then route each story into ChromaDB.

    This is the core of the on-demand flow. Call this when you are ready to
    commit buffered stories to the vector database.

    Response fields
    ---------------
    processed          : stories successfully routed
    narratives_created : new narrative directions created
    narratives_updated : existing narratives updated
    errors             : stories that failed (see server logs for details)
    duration_seconds   : wall time for the entire process call
    buffer_remaining   : stories left in buffer (>0 only if max_stories was hit)
    narratives_touched : list of affected narratives {id, name, action, model_risk, event_count}
    """
    loop = asyncio.get_event_loop()
    start = time.time()

    # Drain up to max_stories; leave the rest in the buffer
    all_buffered = story_buffer.drain()
    to_process = all_buffered[:req.max_stories]
    overflow = all_buffered[req.max_stories:]

    # Put overflow back — drain() is atomic, so we re-add what we didn't use
    if overflow:
        story_buffer.add_batch(overflow)

    if not to_process:
        return {
            "processed": 0,
            "narratives_created": 0,
            "narratives_updated": 0,
            "errors": 0,
            "duration_seconds": 0.0,
            "buffer_remaining": story_buffer.size(),
            "narratives_touched": [],
        }

    # Build text list for batch embedding
    texts = [f"{s.headline}\n\n{s.body}" for s in to_process]

    # One Modal call for all embeddings
    try:
        embeddings = await loop.run_in_executor(None, lambda: embed_batch(texts))
    except Exception as e:
        logger.error(f"embed_batch failed: {e}")
        # Put stories back so they can be retried
        story_buffer.add_batch(to_process)
        raise

    # Route each story with its embedding
    created = updated = errors = 0
    narratives_touched: dict[str, dict] = {}

    for story, embedding in zip(to_process, embeddings):
        try:
            result = await loop.run_in_executor(
                None,
                lambda s=story, emb=embedding: route_with_embedding(s.headline, s.body, emb)
            )
            if result["action"] == "created":
                created += 1
            else:
                updated += 1

            nid = result["narrative_id"]
            narratives_touched[nid] = {
                "id": nid,
                "name": result["narrative_name"],
                "action": result["action"],
                "model_risk": result.get("model_risk"),
                "event_count": result.get("narrative_event_count"),
            }

            # Broadcast to SSE stream if available
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
            errors += 1
            logger.error(f"route_with_embedding failed [{story.headline[:60]}]: {e}")

    return {
        "processed": created + updated,
        "narratives_created": created,
        "narratives_updated": updated,
        "errors": errors,
        "duration_seconds": round(time.time() - start, 2),
        "buffer_remaining": story_buffer.size(),
        "narratives_touched": list(narratives_touched.values()),
    }


# ---------------------------------------------------------------------------
# DELETE /api/pipeline/buffer  — discard buffered stories
# ---------------------------------------------------------------------------

@router.delete("/buffer")
def clear_buffer():
    """
    Discard all stories currently in the buffer without processing them.
    Use this to reset after a bad scrape run or to start fresh.
    """
    count_before = story_buffer.size()
    story_buffer.clear()
    return {"cleared": count_before, "buffer_remaining": 0}


# ---------------------------------------------------------------------------
# GET /api/pipeline/stats  — background auto-pipeline counters
# ---------------------------------------------------------------------------

@router.get("/stats")
def get_pipeline_stats():
    """
    Session-wide counters for the background auto-pipeline.

    These counters only reflect the background loop started at server boot
    (if AUTO_START_PIPELINE=true). Manual scrape and process calls are NOT
    counted here — use the response bodies from those endpoints instead.
    """
    narratives = vector_store.get_all_narratives()
    active = [n for n in narratives if n.is_active]

    total_events = sum(n.event_count for n in narratives)

    return {
        "pipeline": {
            **pipeline_stats,
            "queue_size": story_buffer.size(),
        },
        "narratives": {
            "total": len(narratives),
            "active": len(active),
        },
        "events": {
            "total_events_ingested": total_events,
        },
        "timestamp": time.time(),
    }
