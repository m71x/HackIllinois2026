from fastapi import APIRouter
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
