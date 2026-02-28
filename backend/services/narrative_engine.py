"""
Narrative Routing Engine
========================
Decides whether an incoming story updates an existing narrative direction
or spawns a new one, then persists the result to ChromaDB.

Two entry points:

    ingest_story(headline, body)
        All-in-one: embeds the story itself, then routes. Use for single
        manual ingests where you don't need batch efficiency.

    route_with_embedding(headline, body, embedding)
        Routing only — embedding is supplied externally. Use this when
        the caller has already batch-embedded a set of stories via
        embed_batch() for efficiency (the /api/pipeline/process flow).

Routing rule (cosine distance in [0, 2]):
    distance < NEW_NARRATIVE_THRESHOLD  → update existing narrative
    distance >= NEW_NARRATIVE_THRESHOLD → create new narrative direction
"""

import time
import threading
from core.config import settings
from db import vector_store
from models.narrative import NarrativeDirection
from services.embedder import embed_text
from services.llm_client import score_story, label_narrative

NEW_NARRATIVE_THRESHOLD: float = settings.new_narrative_threshold

# Prevents two threads from simultaneously deciding to create the same narrative
_route_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def ingest_story(headline: str, body: str) -> dict:
    """
    Embed the story, then route it. Convenience wrapper for single-story use.
    Calls embed_text() → Modal Embedder (one network call per story).
    For bulk processing, use route_with_embedding() with pre-computed embeddings.
    """
    full_text = f"{headline}\n\n{body}"
    embedding = embed_text(full_text)
    return route_with_embedding(headline, body, embedding)


def route_with_embedding(headline: str, body: str, embedding: list[float]) -> dict:
    """
    Route a story using a pre-computed embedding vector.

    This is the core routing function. The embedding must be a 384-dim
    L2-normalized float list produced by the Modal Embedder.

    Called by:
        - ingest_story()               (single-story path)
        - /api/pipeline/process        (batch path — embedding pre-computed upstream)
    """
    full_text = f"{headline}\n\n{body}"

    with _route_lock:
        nearest = vector_store.query_nearest(embedding, n_results=5)
        best_narrative, best_distance = (
            (nearest[0][0], nearest[0][1]) if nearest else (None, float("inf"))
        )
        route_to_existing = (
            best_narrative is not None and best_distance < NEW_NARRATIVE_THRESHOLD
        )

    if route_to_existing:
        action = "updated"
        narrative = _update_narrative(best_narrative, embedding, headline, full_text)
    else:
        action = "created"
        narrative = _create_narrative(embedding, headline, full_text)

    return {
        "action": action,
        "narrative_id": narrative.id,
        "narrative_name": narrative.name,
        "best_distance": round(best_distance, 4) if best_narrative else None,
        "threshold": NEW_NARRATIVE_THRESHOLD,
        "current_surprise": narrative.current_surprise,
        "current_impact": narrative.current_impact,
        "model_risk": narrative.model_risk,
        "narrative_event_count": narrative.event_count,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_narrative(
    narrative: NarrativeDirection,
    story_embedding: list[float],
    headline: str,
    full_text: str,
) -> NarrativeDirection:
    scores = score_story(
        story_text=full_text,
        narrative_description=narrative.description,
        existing_surprise=narrative.current_surprise,
        existing_impact=narrative.current_impact,
    )

    now = time.time()
    # Capture event_count BEFORE add_headline increments it (needed for blend)
    n_before = narrative.event_count

    narrative.append_surprise(scores["surprise"], timestamp=now)
    narrative.append_impact(scores["impact"], timestamp=now)
    narrative.add_headline(headline)

    current_embedding = vector_store.get_embedding(narrative.id)
    updated_embedding = _blend_embedding(current_embedding, story_embedding, n=n_before)

    vector_store.update_narrative(narrative, new_embedding=updated_embedding)
    return narrative


def _create_narrative(
    story_embedding: list[float],
    headline: str,
    full_text: str,
) -> NarrativeDirection:
    label = label_narrative(full_text)
    scores = score_story(
        story_text=full_text,
        narrative_description=label["description"],
        existing_surprise=None,
        existing_impact=None,
    )

    now = time.time()
    narrative = NarrativeDirection(
        name=label["name"],
        description=label["description"],
        created_at=now,
        last_updated=now,
    )
    narrative.append_surprise(scores["surprise"], timestamp=now)
    narrative.append_impact(scores["impact"], timestamp=now)
    narrative.add_headline(headline)

    vector_store.add_narrative(narrative, embedding=story_embedding)
    return narrative


def _blend_embedding(
    current: list[float],
    new: list[float],
    n: int,
) -> list[float]:
    """
    Online mean update: new_centroid = old * (n/(n+1)) + story * (1/(n+1))
    n must be the event_count BEFORE the new story was added.
    Re-normalizes to unit length after blending.
    """
    if n <= 0:
        return new
    w_old = n / (n + 1)
    w_new = 1.0 / (n + 1)
    blended = [c * w_old + v * w_new for c, v in zip(current, new)]
    mag = sum(x ** 2 for x in blended) ** 0.5
    return [x / mag for x in blended] if mag > 0 else blended
