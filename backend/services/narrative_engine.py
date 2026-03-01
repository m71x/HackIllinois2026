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
from services.llm_client import label_narrative  # score_story replaced by heuristic

NEW_NARRATIVE_THRESHOLD: float = settings.new_narrative_threshold

# Prevents two threads from simultaneously deciding to create the same narrative
_route_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Fast heuristic scorer — replaces Cerebras score_story() calls
# ---------------------------------------------------------------------------

_HIGH_IMPACT: frozenset[str] = frozenset([
    "collapse", "default", "bankrupt", "crisis", "crash", "recession",
    "shutdown", "sanctions", "war", "emergency", "contagion", "bank run",
    "rate hike", "rate cut", "systemic", "panic", "invasion", "devaluation",
    "insolvency", "failure", "downgrade", "freeze", "seized", "bailout",
    "hyperinflation", "debt ceiling", "sovereign default", "credit crisis",
])

_MED_IMPACT: frozenset[str] = frozenset([
    "inflation", "tariff", "regulation", "earnings", "gdp", "unemployment",
    "deficit", "debt", "risk", "concern", "warning", "fell", "surge",
    "decline", "rally", "threat", "pressure", "layoff", "volatility",
    "correction", "tightening", "slowdown", "contraction", "revision",
    "disruption", "shortage", "interest rate", "yield", "spread",
])

# Explicit linguistic surprise signals — the author themselves flagged it as unexpected
_SHOCK_LANGUAGE: frozenset[str] = frozenset([
    "unexpected", "unexpectedly",
    "surprise", "surprised", "surprises", "surprising",
    "shocking", "shocked", "shockingly",
    "sudden", "suddenly",
    "unprecedented",
    "never before",
    "first time in",
    "reversal", "u-turn",
    "emergency",
    "snap decision",
    "flash crash",
    "panic",
    "abrupt", "abruptly",
])

# Superlative / record language — bigger-than-expected magnitude
_MAGNITUDE_LANGUAGE: frozenset[str] = frozenset([
    "record", "all-time", "all time",
    "biggest", "largest", "worst", "highest", "lowest",
    "most since", "highest since", "lowest since",
    "decade-high", "decade-low", "decade high", "decade low",
    "multi-year", "multiyear",
    "40-year", "50-year", "30-year", "100-year",
    "historic",
    "fastest", "sharpest", "steepest",
])

# Directional tone for inversion detection
_SURGE_TONE: frozenset[str] = frozenset([
    "surges", "surge", "soars", "soar", "rallies", "rally",
    "jumps", "jump", "spikes", "spike", "rises", "rise",
    "beats", "beat", "record high", "boom",
])
_DROP_TONE: frozenset[str] = frozenset([
    "falls", "fall", "drops", "drop", "declines", "decline",
    "slumps", "slump", "plunges", "plunge", "tumbles", "tumble",
    "sinks", "sink", "crashes", "crash", "misses", "miss",
    "selloff", "sell-off", "collapses", "collapse",
])


def _heuristic_score(
    headline: str,
    body: str,
    distance: float,
    narrative: NarrativeDirection | None = None,
) -> dict:
    """
    Multi-signal surprise scorer.  Zero latency, no network calls.

    Surprise is built from five independent signals:

      1. Distance base  — how far the story sits from the nearest narrative
         centroid.  Two-segment scale keeps updates and creates in natural
         sub-ranges, leaving headroom for the correction signals below.
           Updates  (d < threshold): [0.02, 0.60]
           Creates  (d >= threshold): [0.60, 0.85]

      2. Shock language — explicit linguistic markers that the author flagged
         as unexpected ("unprecedented", "sudden", "surprise", "reversal", …).
         Each hit adds +0.08, capped at +0.20.

      3. Magnitude language — superlatives and records signal a bigger-than-
         expected event ("record", "all-time", "40-year high", "fastest ever").
         Each hit adds +0.06, capped at +0.15.

      4. Staleness bonus — a story that re-activates a dormant narrative is
         inherently more surprising than one arriving in an active feed.
         +0.03 (>12h)  →  +0.07 (>48h)  →  +0.12 (>7 days).

      5. Inversion bonus — if the new story's directional tone contradicts the
         narrative's established trend (surge story into a dropping narrative,
         or crash story into a calm one), add +0.08–0.10.

      Maturity dampening: high event_count narratives (well-understood by
      the market) get their distance base dampened by up to 25%.  Explicit
      shock/magnitude signals are NOT dampened — a record event is still
      record regardless of how long the narrative has been tracked.

    Impact — unchanged keyword-count scorer with diminishing returns.
    Range: ~0.04 (no signal) → 1.0 (multiple high-severity hits).
    """
    T = NEW_NARRATIVE_THRESHOLD
    text = (headline + " " + body[:600]).lower()

    # ── 1. Distance base ─────────────────────────────────────────────────────
    if distance < T:
        raw = distance / max(T, 0.01)
        base = 0.02 + 0.58 * (raw ** 0.65)          # [0.02, 0.60]
    else:
        effective = min(distance if distance < float("inf") else 1.5, 2.0)
        raw = (effective - T) / max(2.0 - T, 0.01)
        base = 0.60 + 0.25 * (raw ** 0.5)           # [0.60, 0.85]

    # ── 2. Shock language ────────────────────────────────────────────────────
    shock_hits = sum(1 for kw in _SHOCK_LANGUAGE if kw in text)
    shock_boost = min(0.20, shock_hits * 0.08)

    # ── 3. Magnitude / record language ───────────────────────────────────────
    mag_hits = sum(1 for kw in _MAGNITUDE_LANGUAGE if kw in text)
    mag_boost = min(0.15, mag_hits * 0.06)

    # ── 4. Staleness bonus ───────────────────────────────────────────────────
    stale_boost = 0.0
    if narrative is not None:
        age_hours = (time.time() - narrative.last_updated) / 3600.0
        if age_hours > 168:    # > 7 days dormant
            stale_boost = 0.12
        elif age_hours > 48:   # > 2 days
            stale_boost = 0.07
        elif age_hours > 12:   # > 12 hours
            stale_boost = 0.03

    # ── 5. Tone inversion ────────────────────────────────────────────────────
    inversion_boost = 0.0
    if narrative is not None and narrative.current_surprise is not None:
        established = narrative.current_surprise   # EMA over last 50 points
        story_surge = any(kw in text for kw in _SURGE_TONE)
        story_drop  = any(kw in text for kw in _DROP_TONE)
        # Calm narrative suddenly re-escalates with crisis/shock tone
        if established < 0.35 and story_drop and sum(1 for kw in _HIGH_IMPACT if kw in text) >= 1:
            inversion_boost = 0.10
        # High-tension narrative suddenly shows a resolution/drop signal
        elif established > 0.65 and story_surge and not story_drop:
            inversion_boost = 0.08

    # ── Maturity dampening on the distance base only ─────────────────────────
    maturity = 1.0
    if narrative is not None:
        n = narrative.event_count
        if n > 100:
            maturity = 0.75
        elif n > 50:
            maturity = 0.88
        elif n > 20:
            maturity = 0.94

    surprise_raw = (base * maturity) + shock_boost + mag_boost + stale_boost + inversion_boost
    surprise = round(min(1.0, max(0.02, surprise_raw)), 3)

    # ── Impact (unchanged) ───────────────────────────────────────────────────
    high_hits = sum(1 for kw in _HIGH_IMPACT if kw in text)
    med_hits  = sum(1 for kw in _MED_IMPACT  if kw in text)
    impact_from_high = min(0.75, high_hits * 0.30)
    impact_from_med  = min(0.25, med_hits  * 0.04)
    impact = round(max(0.04, impact_from_high + impact_from_med), 3)

    return {"surprise": surprise, "impact": impact}


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
        best_narrative, best_distance, best_emb = (
            (nearest[0][0], nearest[0][1], nearest[0][2]) if nearest else (None, float("inf"), None)
        )
        route_to_existing = (
            best_narrative is not None and best_distance < NEW_NARRATIVE_THRESHOLD
        )

    if route_to_existing:
        action = "updated"
        assert best_narrative is not None
        narrative = _update_narrative(
            best_narrative, embedding, headline, full_text, best_distance,
            current_embedding=best_emb,
        )
    else:
        action = "created"
        narrative = _create_narrative(embedding, headline, full_text, best_distance)

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


# Batch size for bulk routing — one ChromaDB query per chunk instead of per story.
_ROUTE_BATCH_SIZE = 256


def batch_query_nearest(
    embeddings: list[list[float]],
) -> list[tuple[object, float, list[float]] | None]:
    """
    Query ChromaDB for the nearest narrative of every embedding in one
    batched round-trip per _ROUTE_BATCH_SIZE stories (instead of one
    round-trip per story).

    Returns one entry per input embedding:
        (NarrativeDirection, distance, stored_embedding)  if a match exists
        None                                               if collection is empty

    Acquires _route_lock once per chunk of _ROUTE_BATCH_SIZE.
    """
    results: list[tuple[object, float, list[float]] | None] = []
    for i in range(0, len(embeddings), _ROUTE_BATCH_SIZE):
        chunk = embeddings[i : i + _ROUTE_BATCH_SIZE]
        with _route_lock:
            batch = vector_store.query_nearest_batch(chunk, n_results=1)
        for row in batch:
            if row:
                results.append((row[0][0], row[0][1], row[0][2]))
            else:
                results.append(None)
    return results


def route_with_precomputed_nearest(
    headline: str,
    body: str,
    embedding: list[float],
    nearest: tuple[object, float, list[float]] | None,
) -> dict:
    """
    Route a story using a pre-computed nearest-narrative result.

    Use this in the bulk ingest path where batch_query_nearest() has already
    done all ChromaDB queries.  Skips the query entirely for the UPDATE path.

    For the CREATE path: re-queries ChromaDB live (under _route_lock) before
    creating, so a story can route to a narrative created by another story in
    the same batch since the batch snapshot was taken.
    """
    full_text = f"{headline}\n\n{body}"

    if nearest is not None:
        best_narrative, best_distance, best_emb = nearest
        route_to_existing = best_distance < NEW_NARRATIVE_THRESHOLD
    else:
        best_narrative, best_distance, best_emb = None, float("inf"), None
        route_to_existing = False

    if route_to_existing:
        action = "updated"
        narrative = _update_narrative(
            best_narrative, embedding, headline, full_text, best_distance,  # type: ignore[arg-type]
            current_embedding=best_emb,
        )
    else:
        # Re-query live ChromaDB before creating — another story in the same
        # batch may have already created a matching narrative since the batch
        # snapshot was taken.
        with _route_lock:
            live = vector_store.query_nearest(embedding, n_results=1)
            if live and live[0][1] < NEW_NARRATIVE_THRESHOLD:
                best_narrative, best_distance, best_emb = live[0]
                route_to_existing = True

        if route_to_existing:
            action = "updated"
            narrative = _update_narrative(
                best_narrative, embedding, headline, full_text, best_distance,  # type: ignore[arg-type]
                current_embedding=best_emb,
            )
        else:
            action = "created"
            narrative = _create_narrative(embedding, headline, full_text, best_distance)

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
    distance: float = 0.2,
    current_embedding: list[float] | None = None,
) -> NarrativeDirection:
    # Heuristic scorer — passes narrative for staleness/inversion/maturity signals
    scores = _heuristic_score(headline, full_text, distance, narrative=narrative)

    now = time.time()
    # Capture event_count BEFORE add_headline increments it (needed for blend)
    n_before = narrative.event_count

    narrative.append_surprise(scores["surprise"], timestamp=now)
    narrative.append_impact(scores["impact"], timestamp=now)
    narrative.add_headline(headline)

    # Use pre-fetched embedding if provided (avoids a separate get_embedding() call)
    if current_embedding is None:
        current_embedding = vector_store.get_embedding(narrative.id)
    updated_embedding = _blend_embedding(current_embedding, story_embedding, n=n_before)

    vector_store.update_narrative(narrative, new_embedding=updated_embedding)
    return narrative


def _create_narrative(
    story_embedding: list[float],
    headline: str,
    full_text: str,
    distance: float = float("inf"),
) -> NarrativeDirection:
    label = label_narrative(full_text)
    scores = _heuristic_score(headline, full_text, distance)

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
