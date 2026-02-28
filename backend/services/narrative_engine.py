"""
Core routing logic: decides whether an incoming news story updates an
existing narrative direction or spawns a new one.

Decision rule:
  best_cosine_distance < NEW_NARRATIVE_THRESHOLD  → update existing narrative
  best_cosine_distance >= NEW_NARRATIVE_THRESHOLD → create new narrative

Cosine distance is in [0, 2]:
  0.0  = identical vectors
  1.0  = orthogonal (unrelated)
  2.0  = opposite directions

A threshold around 0.35–0.45 works well for news narratives — tight enough
to keep distinct topics separate, loose enough to cluster related stories.
"""

import time
from core.config import settings
from db import vector_store
from models.narrative import NarrativeDirection
from services.embedder import embed_text
from services.cerebras_client import score_story, label_narrative


NEW_NARRATIVE_THRESHOLD: float = settings.new_narrative_threshold


def ingest_story(headline: str, body: str) -> dict:
    """
    Main entry point. Takes a raw news story, routes it to the correct
    narrative direction (or creates a new one), and updates metrics.

    Returns a summary of what happened.
    """
    full_text = f"{headline}\n\n{body}"
    story_embedding = embed_text(full_text)

    nearest = vector_store.query_nearest(story_embedding, n_results=5)

    if nearest:
        best_narrative, best_distance = nearest[0]
    else:
        best_narrative, best_distance = None, float("inf")

    # --- Route: update vs. create ---
    if best_narrative is not None and best_distance < NEW_NARRATIVE_THRESHOLD:
        action = "updated"
        narrative = _update_narrative(best_narrative, story_embedding, headline, full_text)
    else:
        action = "created"
        narrative = _create_narrative(story_embedding, headline, full_text)

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
    narrative.append_surprise(scores["surprise"], timestamp=now)
    narrative.append_impact(scores["impact"], timestamp=now)
    narrative.add_headline(headline)

    # Rolling average embedding: blend story vector into narrative centroid
    updated_embedding = _blend_embedding(
        current=vector_store.collection.get(
            ids=[narrative.id], include=["embeddings"]
        )["embeddings"][0],
        new=story_embedding,
        n=narrative.event_count,
    )

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


def _blend_embedding(current: list[float], new: list[float], n: int) -> list[float]:
    """
    Incrementally update the narrative centroid with a new story vector.
    Uses online mean: centroid_new = centroid_old * (n/(n+1)) + new * (1/(n+1))
    """
    weight_old = n / (n + 1)
    weight_new = 1 / (n + 1)
    return [c * weight_old + v * weight_new for c, v in zip(current, new)]
