from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db import vector_store
from services.embedder import embed_text

router = APIRouter()


@router.get("")
def list_narratives():
    narratives = vector_store.get_all_narratives()
    return [
        {
            "id": n.id,
            "name": n.name,
            "description": n.description,
            "event_count": n.event_count,
            "current_surprise": n.current_surprise,
            "current_impact": n.current_impact,
            "model_risk": n.model_risk,
            "last_updated": n.last_updated,
        }
        for n in narratives
    ]


@router.get("/{narrative_id}")
def get_narrative(narrative_id: str):
    n = vector_store.get_narrative(narrative_id)
    if not n:
        raise HTTPException(status_code=404, detail="Narrative not found")
    return n.model_dump()


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


@router.post("/search")
def search_narratives(req: SearchRequest):
    embedding = embed_text(req.query)
    results = vector_store.query_nearest(embedding, n_results=req.n_results)
    return [
        {
            "narrative": n.model_dump(),
            "distance": round(d, 4),
        }
        for n, d in results
    ]
