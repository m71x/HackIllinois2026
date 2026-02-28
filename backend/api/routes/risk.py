import time
from fastapi import APIRouter, Query
from db import vector_store

router = APIRouter()


@router.get("")
def get_risk_index():
    """
    Aggregate Real-World Model Risk Index across all active narratives.
    Returns the index value and per-narrative breakdown.
    """
    narratives = vector_store.get_all_narratives()
    if not narratives:
        return {"model_risk_index": None, "narrative_count": 0, "breakdown": []}

    risks = [n.model_risk for n in narratives if n.model_risk is not None]
    index = max(risks) if risks else None  # use max: any high-risk narrative elevates overall risk

    breakdown = sorted(
        [
            {
                "id": n.id,
                "name": n.name,
                "surprise": n.current_surprise,
                "impact": n.current_impact,
                "model_risk": n.model_risk,
                "event_count": n.event_count,
            }
            for n in narratives
            if n.model_risk is not None
        ],
        key=lambda x: x["model_risk"],
        reverse=True,
    )

    return {
        "model_risk_index": round(index, 4) if index is not None else None,
        "narrative_count": len(narratives),
        "breakdown": breakdown,
    }


@router.get("/history")
def get_risk_history(
    window: int = Query(24, description="Hours of history to return"),
    resolution: int = Query(100, description="Max number of data points"),
):
    """
    Get historical risk index values for charting.
    Aggregates risk from all narratives over time.
    """
    narratives = vector_store.get_all_narratives()

    if not narratives:
        return {"history": []}

    # Collect all timestamps with risk values
    all_points = []
    cutoff = time.time() - (window * 3600)

    for n in narratives:
        for i, sp in enumerate(n.surprise_series):
            if sp.timestamp < cutoff:
                continue
            if i < len(n.impact_series):
                risk = (sp.value * n.impact_series[i].value) ** 0.5
                all_points.append({"timestamp": sp.timestamp, "model_risk_index": risk})

    if not all_points:
        # Return synthetic data if no real data exists
        now = time.time()
        return {
            "history": [
                {"timestamp": now - (window - i) * 3600, "model_risk_index": 0.5}
                for i in range(min(24, window))
            ]
        }

    # Sort by timestamp and downsample if needed
    all_points.sort(key=lambda x: x["timestamp"])

    if len(all_points) > resolution:
        step = len(all_points) // resolution
        all_points = all_points[::step][:resolution]

    return {"history": all_points}
