from fastapi import APIRouter
from pydantic import BaseModel
from services.narrative_engine import ingest_story
from core.state import pipeline_stats, broadcast_event

router = APIRouter()


class IngestRequest(BaseModel):
    headline: str
    body: str = ""


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
