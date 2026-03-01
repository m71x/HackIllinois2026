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
    Cluster all narrative embeddings via K-means++ (cosine), then project each
    cluster centroid to 2D via PCA.  Returns one node per cluster and edges
    between semantically related clusters.
    """
    try:
        import numpy as np
        import math
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
    X = np.array(emb_list, dtype=np.float32)  # (N, 384), L2-normalized
    N = X.shape[0]

    # ── Auto K selection ─────────────────────────────────────────────────────
    # Targets roughly sqrt(N) clusters, bounded to [3, 12]
    k = max(3, min(12, int(math.sqrt(N))))
    if N <= k:
        k = N

    # ── K-means++ (cosine) ───────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    # Seeding: spread initial centroids using k-means++ distance weighting
    seed_idx = [int(rng.integers(N))]
    for _ in range(k - 1):
        dists = 1.0 - X @ X[seed_idx].T          # cosine distance to chosen seeds
        min_dists = dists.min(axis=1).clip(0)
        probs = min_dists / (min_dists.sum() + 1e-10)
        seed_idx.append(int(rng.choice(N, p=probs)))

    centroids = X[seed_idx].copy()
    labels = np.zeros(N, dtype=int)

    for _ in range(80):
        sims = X @ centroids.T                    # (N, k) cosine similarity
        new_labels = np.argmax(sims, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                c = X[mask].mean(axis=0)
                n = float(np.linalg.norm(c))
                centroids[j] = c / n if n > 0 else c

    # ── Build cluster metadata ────────────────────────────────────────────────
    cluster_nodes_raw = []
    for j in range(k):
        member_mask = labels == j
        if not member_mask.any():
            continue
        members = [narr_list[i] for i in range(N) if member_mask[i]]
        member_embs = X[member_mask]
        centroid = centroids[j]

        # Representative narrative = the one whose embedding is closest to centroid
        sims_to_centroid = member_embs @ centroid
        rep_name = members[int(np.argmax(sims_to_centroid))].name

        cluster_nodes_raw.append({
            "_j": j,
            "_centroid": centroid,
            "id": f"cluster_{j}",
            "label": rep_name,
            "member_count": int(member_mask.sum()),
            "total_events": int(sum(m.event_count for m in members)),
            "model_risk": round(float(np.mean([m.model_risk or 0 for m in members])), 4),
            "current_surprise": round(float(np.mean([m.current_surprise or 0 for m in members])), 4),
            "current_impact": round(float(np.mean([m.current_impact or 0 for m in members])), 4),
            "member_names": [m.name for m in members[:6]],
        })

    if not cluster_nodes_raw:
        return {"nodes": [], "edges": []}

    # ── PCA of cluster centroids → 2D ────────────────────────────────────────
    C = np.array([cn["_centroid"] for cn in cluster_nodes_raw], dtype=np.float32)
    NC = C.shape[0]

    if NC == 1:
        x2d = np.zeros((1, 2), dtype=np.float32)
    else:
        C_c = C - C.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(C_c, full_matrices=False)
            x2d = C_c @ Vt[:2].T
        except Exception:
            x2d = np.zeros((NC, 2), dtype=np.float32)

        for d in range(2):
            col = x2d[:, d]
            lo, hi = float(col.min()), float(col.max())
            if hi > lo:
                x2d[:, d] = (col - lo) / (hi - lo) * 2.0 - 1.0

    # ── Inter-cluster edges ───────────────────────────────────────────────────
    sim_matrix = (C @ C.T).astype(float)
    np.fill_diagonal(sim_matrix, -1.0)

    STRONG_THRESHOLD = 0.45   # above this → solid edge
    MAX_EDGES_PER_NODE = 3

    edge_set: set[tuple[int, int]] = set()
    edges = []

    # Phase 1: Maximum spanning tree (Kruskal) — guarantees every node has ≥1 edge.
    # We sort all pairs by similarity descending and union-find to build the MST.
    all_pairs = sorted(
        ((float(sim_matrix[i, j]), i, j) for i in range(NC) for j in range(i + 1, NC)),
        reverse=True,
    )
    parent = list(range(NC))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for sim, i, j in all_pairs:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj
            key = (min(i, j), max(i, j))
            edge_set.add(key)
            edges.append({
                "source": cluster_nodes_raw[i]["id"],
                "target": cluster_nodes_raw[j]["id"],
                "similarity": round(sim, 4),
                "weak": sim < STRONG_THRESHOLD,   # bridge edge flag for frontend styling
            })

    # Phase 2: Add extra strong edges (top-3 per node above threshold, deduplicated).
    for i in range(NC):
        top_idxs = np.argsort(sim_matrix[i])[-MAX_EDGES_PER_NODE:][::-1]
        for j_idx in top_idxs:
            sim = float(sim_matrix[i, j_idx])
            if sim < STRONG_THRESHOLD:
                continue
            key = (min(i, j_idx), max(i, j_idx))
            if key not in edge_set:
                edge_set.add(key)
                edges.append({
                    "source": cluster_nodes_raw[i]["id"],
                    "target": cluster_nodes_raw[j_idx]["id"],
                    "similarity": round(sim, 4),
                    "weak": False,
                })

    # ── Finalize (strip internal fields) ─────────────────────────────────────
    nodes = [
        {
            "id": cn["id"],
            "label": cn["label"],
            "member_count": cn["member_count"],
            "total_events": cn["total_events"],
            "model_risk": cn["model_risk"],
            "current_surprise": cn["current_surprise"],
            "current_impact": cn["current_impact"],
            "member_names": cn["member_names"],
            "x": round(float(x2d[i, 0]), 4),
            "y": round(float(x2d[i, 1]), 4),
        }
        for i, cn in enumerate(cluster_nodes_raw)
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
            for n, d, _emb in results
        ]
    }
