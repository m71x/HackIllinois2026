from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from db import vector_store
from services.embedder import embed_text

router = APIRouter()


def _compute_trend(series: list) -> str:
    """Compute trend from recent values in a time series."""
    if len(series) < 2:
        return "stable"
    recent = [p.value for p in series[-5:]]
    if len(recent) < 2:
        return "stable"
    diff = recent[-1] - recent[0]
    if diff > 0.05:
        return "rising"
    elif diff < -0.05:
        return "falling"
    return "stable"


@router.get("")
def list_narratives(
    active_only: bool = Query(False),
    sort_by: str = Query("recency"),
    limit: int = Query(50),
):
    narratives = vector_store.get_all_narratives()

    # Build response objects
    result = []
    for n in narratives:
        result.append({
            "id": n.id,
            "name": n.name,
            "description": n.description,
            "event_count": n.event_count,
            "current_surprise": n.current_surprise,
            "current_impact": n.current_impact,
            "model_risk": n.model_risk,
            "last_updated": n.last_updated,
            "surprise_trend": _compute_trend(n.surprise_series),
        })

    # Sort based on sort_by parameter
    if sort_by == "risk":
        result.sort(key=lambda x: x["model_risk"] or 0, reverse=True)
    elif sort_by == "events":
        result.sort(key=lambda x: x["event_count"], reverse=True)
    else:  # recency
        result.sort(key=lambda x: x["last_updated"], reverse=True)

    # Apply limit
    result = result[:limit]

    return {"narratives": result}


@router.get("/{narrative_id}")
def get_narrative(narrative_id: str):
    n = vector_store.get_narrative(narrative_id)
    if not n:
        raise HTTPException(status_code=404, detail="Narrative not found")
    return n.model_dump()


@router.get("/{narrative_id}/history")
def get_narrative_history(narrative_id: str):
    """Get narrative with full time series data for charts."""
    n = vector_store.get_narrative(narrative_id)
    if not n:
        raise HTTPException(status_code=404, detail="Narrative not found")

    return {
        "id": n.id,
        "name": n.name,
        "description": n.description,
        "event_count": n.event_count,
        "model_risk": n.model_risk,
        "last_updated": n.last_updated,
        "recent_headlines": n.recent_headlines,
        "surprise_series": [{"timestamp": p.timestamp, "value": p.value} for p in n.surprise_series],
        "impact_series": [{"timestamp": p.timestamp, "value": p.value} for p in n.impact_series],
        "model_risk_series": [
            {"timestamp": p.timestamp, "value": (p.value * n.impact_series[i].value) ** 0.5}
            for i, p in enumerate(n.surprise_series)
            if i < len(n.impact_series)
        ],
    }


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


@router.post("/search")
def search_narratives(req: SearchRequest):
    embedding = embed_text(req.query)
    results = vector_store.query_nearest(embedding, n_results=req.n_results)

    # Convert distance to similarity (cosine distance is 0-2, so similarity = 1 - distance/2)
    return {
        "results": [
            {
                "narrative": {
                    "id": n.id,
                    "name": n.name,
                    "description": n.description,
                    "model_risk": n.model_risk,
                    "event_count": n.event_count,
                },
                "similarity": round(1 - d / 2, 4),
            }
            for n, d in results
        ]
    }
