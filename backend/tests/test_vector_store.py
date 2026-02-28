"""Tests for db/vector_store.py using an in-memory ChromaDB collection."""

import math
import time
import pytest
from models.narrative import NarrativeDirection


def make_unit_vector(seed: int, dim: int = 384) -> list[float]:
    v = [math.sin(seed + i * 0.01) for i in range(dim)]
    mag = sum(x ** 2 for x in v) ** 0.5
    return [x / mag for x in v]


def make_narrative(name: str = "Test", description: str = "A test narrative") -> NarrativeDirection:
    n = NarrativeDirection(name=name, description=description)
    n.append_surprise(0.5)
    n.append_impact(0.4)
    n.add_headline("Test headline")
    return n


class TestAddAndGet:
    def test_add_and_retrieve(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative("Fed policy")
        embedding = make_unit_vector(1)
        vs.add_narrative(n, embedding=embedding)

        retrieved = vs.get_narrative(n.id)
        assert retrieved is not None
        assert retrieved.id == n.id
        assert retrieved.name == "Fed policy"

    def test_get_nonexistent_returns_none(self, patched_vector_store):
        vs = patched_vector_store
        assert vs.get_narrative("nonexistent-id-12345") is None

    def test_narrative_count_increments(self, patched_vector_store):
        vs = patched_vector_store
        assert vs.narrative_count() == 0
        vs.add_narrative(make_narrative("N1"), embedding=make_unit_vector(1))
        vs.add_narrative(make_narrative("N2"), embedding=make_unit_vector(2))
        assert vs.narrative_count() == 2

    def test_get_all_narratives(self, patched_vector_store):
        vs = patched_vector_store
        for i in range(4):
            vs.add_narrative(make_narrative(f"Narrative {i}"), embedding=make_unit_vector(i))
        all_n = vs.get_all_narratives()
        assert len(all_n) == 4
        names = {n.name for n in all_n}
        assert names == {"Narrative 0", "Narrative 1", "Narrative 2", "Narrative 3"}

    def test_get_all_empty(self, patched_vector_store):
        assert patched_vector_store.get_all_narratives() == []


class TestUpdate:
    def test_update_name(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative("Old name")
        vs.add_narrative(n, embedding=make_unit_vector(1))

        n.name = "New name"
        vs.update_narrative(n)

        retrieved = vs.get_narrative(n.id)
        assert retrieved.name == "New name"

    def test_update_time_series(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        vs.add_narrative(n, embedding=make_unit_vector(1))

        n.append_surprise(0.9)
        vs.update_narrative(n)

        retrieved = vs.get_narrative(n.id)
        assert retrieved.current_surprise == pytest.approx(0.9)

    def test_update_with_new_embedding(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        vs.add_narrative(n, embedding=make_unit_vector(1))

        new_embedding = make_unit_vector(99)
        vs.update_narrative(n, new_embedding=new_embedding)

        stored_emb = vs.get_embedding(n.id)
        assert len(stored_emb) == 384


class TestGetEmbedding:
    def test_returns_correct_dimension(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        emb = make_unit_vector(42)
        vs.add_narrative(n, embedding=emb)
        stored = vs.get_embedding(n.id)
        assert len(stored) == 384

    def test_returns_stored_vector(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        emb = make_unit_vector(7)
        vs.add_narrative(n, embedding=emb)
        stored = vs.get_embedding(n.id)
        assert all(abs(a - b) < 1e-6 for a, b in zip(emb, stored))

    def test_missing_id_returns_empty(self, patched_vector_store):
        result = patched_vector_store.get_embedding("bad-id")
        assert result == []


class TestDelete:
    def test_delete_removes_narrative(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        vs.add_narrative(n, embedding=make_unit_vector(1))
        vs.delete_narrative(n.id)
        assert vs.get_narrative(n.id) is None

    def test_delete_returns_true(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        vs.add_narrative(n, embedding=make_unit_vector(1))
        assert vs.delete_narrative(n.id) is True

    def test_delete_decrements_count(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        vs.add_narrative(n, embedding=make_unit_vector(1))
        vs.delete_narrative(n.id)
        assert vs.narrative_count() == 0


class TestQueryNearest:
    def test_empty_collection_returns_empty(self, patched_vector_store):
        results = patched_vector_store.query_nearest(make_unit_vector(1), n_results=5)
        assert results == []

    def test_returns_closest_first(self, patched_vector_store):
        vs = patched_vector_store
        # n1 is very close to the query vector (seed=1)
        # n2 is far from it (seed=100)
        n1 = make_narrative("Close narrative")
        n2 = make_narrative("Far narrative")
        vs.add_narrative(n1, embedding=make_unit_vector(1))
        vs.add_narrative(n2, embedding=make_unit_vector(100))

        query = make_unit_vector(1)  # identical to n1
        results = vs.query_nearest(query, n_results=2)

        assert len(results) == 2
        closest_name = results[0][0].name
        assert closest_name == "Close narrative"

    def test_distance_is_near_zero_for_identical(self, patched_vector_store):
        vs = patched_vector_store
        n = make_narrative()
        emb = make_unit_vector(5)
        vs.add_narrative(n, embedding=emb)

        results = vs.query_nearest(emb, n_results=1)
        assert len(results) == 1
        distance = results[0][1]
        assert distance < 1e-4   # cosine distance ~0 for identical vectors

    def test_n_results_capped_by_collection_size(self, patched_vector_store):
        vs = patched_vector_store
        for i in range(3):
            vs.add_narrative(make_narrative(f"N{i}"), embedding=make_unit_vector(i))
        # Asking for 10 but only 3 exist
        results = vs.query_nearest(make_unit_vector(0), n_results=10)
        assert len(results) == 3

    def test_serialization_roundtrip(self, patched_vector_store):
        """Verify time series survives the Chroma serialize/deserialize cycle."""
        vs = patched_vector_store
        n = make_narrative()
        n.append_surprise(0.3)
        n.append_surprise(0.6)
        n.append_impact(0.4)
        n.append_impact(0.8)
        vs.add_narrative(n, embedding=make_unit_vector(1))

        retrieved = vs.get_narrative(n.id)
        assert len(retrieved.surprise_series) == 3   # initial + 2 appends
        assert len(retrieved.impact_series) == 3
        assert retrieved.current_surprise == pytest.approx(0.6)
        assert retrieved.current_impact == pytest.approx(0.8)
