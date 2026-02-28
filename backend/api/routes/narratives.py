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


@router.get("/graph")
def get_narrative_graph():
    """
    Project all narrative embeddings to 2D via PCA.
    Returns nodes with x/y coords, metadata, and similarity edges for graph visualization.
    """
    try:
        import numpy as np
    except ImportError:
        return {"nodes": [], "edges": [], "error": "numpy not available"}

    narratives = vector_store.get_all_narratives()
    if not narratives:
        return {"nodes": [], "edges": []}

    # Fetch embeddings; skip any that are missing/corrupt
    pairs = []
    for n in narratives:
        emb = vector_store.get_embedding(n.id)
        if emb and len(emb) == 384:
            pairs.append((n, emb))

    if not pairs:
        return {"nodes": [], "edges": []}

    narr_list, emb_list = zip(*pairs)
    X = np.array(emb_list, dtype=np.float32)  # (N, 384)
    N = X.shape[0]

    # ── PCA to 2D ────────────────────────────────────────────────────────────
    if N == 1:
        x2d = np.zeros((1, 2), dtype=np.float32)
    else:
        X_c = X - X.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
            x2d = X_c @ Vt[:2].T  # project onto first 2 PCs → (N, 2)
        except Exception:
            x2d = np.zeros((N, 2), dtype=np.float32)

        # Normalize each axis to [-1, 1]
        for d in range(2):
            col = x2d[:, d]
            lo, hi = float(col.min()), float(col.max())
            if hi > lo:
                x2d[:, d] = (col - lo) / (hi - lo) * 2.0 - 1.0

    # ── Pairwise cosine similarity → edges ───────────────────────────────────
    # X rows are L2-normalized (stored that way by the Modal embedder),
    # so dot(a, b) == cosine similarity.
    sim_matrix = (X @ X.T).astype(float)
    np.fill_diagonal(sim_matrix, -1.0)   # exclude self-edges

    EDGE_THRESHOLD = 0.55       # only draw clearly related pairs
    MAX_EDGES_PER_NODE = 3

    edge_set: set[tuple[int, int]] = set()
    edges = []
    for i in range(N):
        top_idxs = np.argsort(sim_matrix[i])[-MAX_EDGES_PER_NODE:][::-1]
        for j in top_idxs:
            sim = float(sim_matrix[i, j])
            if sim < EDGE_THRESHOLD:
                continue
            key = (min(i, j), max(i, j))
            if key not in edge_set:
                edge_set.add(key)
                edges.append({
                    "source": narr_list[i].id,
                    "target": narr_list[j].id,
                    "similarity": round(sim, 4),
                })

    # ── Build node list ───────────────────────────────────────────────────────
    nodes = [
        {
            "id": n.id,
            "name": n.name,
            "x": round(float(x2d[i, 0]), 4),
            "y": round(float(x2d[i, 1]), 4),
            "model_risk": n.model_risk,
            "current_surprise": n.current_surprise,
            "current_impact": n.current_impact,
            "event_count": n.event_count,
            "last_updated": n.last_updated,
        }
        for i, n in enumerate(narr_list)
    ]

    return {"nodes": nodes, "edges": edges}


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
