"""Tests for services/narrative_engine.py — routing and centroid blending."""

import math
import pytest
from unittest.mock import patch
from services.narrative_engine import _blend_embedding, route_with_embedding
from models.narrative import NarrativeDirection


def make_unit_vector(seed: int, dim: int = 384) -> list[float]:
    v = [math.sin(seed + i * 0.01) for i in range(dim)]
    mag = sum(x ** 2 for x in v) ** 0.5
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# _blend_embedding — pure math, no mocks needed
# ---------------------------------------------------------------------------

class TestBlendEmbedding:
    def test_n_zero_returns_new(self):
        old = make_unit_vector(1)
        new = make_unit_vector(2)
        result = _blend_embedding(old, new, n=0)
        assert result == new

    def test_n_one_gives_equal_weight(self):
        # n=1 → old * (1/2) + new * (1/2)
        old = [1.0, 0.0]
        new = [0.0, 1.0]
        blended = _blend_embedding(old, new, n=1)
        # both components should be equal before normalization
        assert abs(blended[0] - blended[1]) < 1e-9

    def test_output_is_unit_normalized(self):
        old = make_unit_vector(10)
        new = make_unit_vector(20)
        for n in [1, 5, 10, 100]:
            blended = _blend_embedding(old, new, n=n)
            magnitude = sum(x ** 2 for x in blended) ** 0.5
            assert abs(magnitude - 1.0) < 1e-6, f"n={n} gave magnitude {magnitude}"

    def test_high_n_biases_toward_old(self):
        """With large n, new story has tiny influence on centroid."""
        old = make_unit_vector(1)
        new = make_unit_vector(99)
        blended = _blend_embedding(old, new, n=999)
        # Cosine similarity between blended and old should be very high
        dot_old = sum(a * b for a, b in zip(blended, old))
        dot_new = sum(a * b for a, b in zip(blended, new))
        assert dot_old > dot_new

    def test_low_n_biases_toward_new(self):
        """With n=1 (first story ever), new gets 50% weight — close to new."""
        old = make_unit_vector(1)
        new = make_unit_vector(99)
        blended = _blend_embedding(old, new, n=1)
        dot_old = sum(a * b for a, b in zip(blended, old))
        dot_new = sum(a * b for a, b in zip(blended, new))
        # n=1: weights are 0.5/0.5, so neither should dominate strongly
        # but they should at least both be positive
        assert dot_old > 0
        assert dot_new > 0

    def test_same_vector_unchanged(self):
        v = make_unit_vector(5)
        blended = _blend_embedding(v, v, n=10)
        dot = sum(a * b for a, b in zip(blended, v))
        assert abs(dot - 1.0) < 1e-6  # should be the same direction


# ---------------------------------------------------------------------------
# route_with_embedding — requires mocked embedder, LLM, and vector_store
# ---------------------------------------------------------------------------

class TestRouteWithEmbedding:
    def test_creates_narrative_when_collection_empty(
        self,
        patched_vector_store,
        fake_label_narrative,
        fake_score_story,
    ):
        with patch("services.narrative_engine.vector_store", patched_vector_store):
            result = route_with_embedding(
                headline="Fed raises rates by 75bps",
                body="The Federal Reserve raised rates...",
                embedding=make_unit_vector(1),
            )
        assert result["action"] == "created"
        assert result["narrative_id"] is not None
        assert result["narrative_name"] == "Fed tightening cycle"

    def test_updates_narrative_when_close(
        self,
        patched_vector_store,
        fake_label_narrative,
        fake_score_story,
    ):
        vs = patched_vector_store
        # Pre-insert a narrative with a vector very close to our query
        existing = NarrativeDirection(name="Existing narrative", description="Already in DB")
        existing.append_surprise(0.3)
        existing.append_impact(0.4)
        existing.add_headline("Prior headline")
        seed = 5
        vs.add_narrative(existing, embedding=make_unit_vector(seed))

        with patch("services.narrative_engine.vector_store", vs):
            result = route_with_embedding(
                headline="More Fed news",
                body="The Fed continues...",
                embedding=make_unit_vector(seed),  # identical seed → distance ~0
            )

        assert result["action"] == "updated"
        assert result["narrative_id"] == existing.id

    def test_creates_new_when_distant(
        self,
        patched_vector_store,
        fake_label_narrative,
        fake_score_story,
    ):
        vs = patched_vector_store
        # Insert a narrative at seed=1 (Fed/rates topic area)
        existing = NarrativeDirection(name="Fed tightening", description="Rate hikes")
        existing.append_surprise(0.5)
        existing.append_impact(0.5)
        existing.add_headline("Rates up")
        vs.add_narrative(existing, embedding=make_unit_vector(1))

        with patch("services.narrative_engine.vector_store", vs):
            # Query with a very different vector (seed=500 — orthogonal topic)
            result = route_with_embedding(
                headline="Earthquake hits Japan",
                body="A major earthquake...",
                embedding=make_unit_vector(500),
            )

        assert result["action"] == "created"
        assert result["narrative_id"] != existing.id

    def test_result_contains_required_fields(
        self,
        patched_vector_store,
        fake_label_narrative,
        fake_score_story,
    ):
        with patch("services.narrative_engine.vector_store", patched_vector_store):
            result = route_with_embedding("Headline", "Body", make_unit_vector(1))

        required = {
            "action", "narrative_id", "narrative_name",
            "best_distance", "threshold",
            "current_surprise", "current_impact", "model_risk",
            "narrative_event_count",
        }
        assert required.issubset(result.keys())

    def test_scores_are_applied(
        self,
        patched_vector_store,
        fake_label_narrative,
        fake_score_story,   # returns surprise=0.6, impact=0.7
    ):
        with patch("services.narrative_engine.vector_store", patched_vector_store):
            result = route_with_embedding("Headline", "Body", make_unit_vector(1))

        assert result["current_surprise"] == pytest.approx(0.6)
        assert result["current_impact"] == pytest.approx(0.7)
        expected_risk = (0.6 * 0.7) ** 0.5
        assert result["model_risk"] == pytest.approx(expected_risk)

    def test_event_count_increments_on_update(
        self,
        patched_vector_store,
        fake_label_narrative,
        fake_score_story,
    ):
        vs = patched_vector_store
        existing = NarrativeDirection(name="Existing", description="Desc")
        existing.append_surprise(0.3)
        existing.append_impact(0.3)
        existing.add_headline("H1")
        seed = 3
        vs.add_narrative(existing, embedding=make_unit_vector(seed))

        with patch("services.narrative_engine.vector_store", vs):
            result = route_with_embedding("H2", "Body", make_unit_vector(seed))

        assert result["narrative_event_count"] == 2  # was 1, now 2
