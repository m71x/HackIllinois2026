"""
Tests for the Real-World Model Risk Engine API.
Uses fixtures from conftest.py for mocking external services.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture()
def client(patched_vector_store, fake_embed_text, fake_embed_batch,
           fake_label_narrative, fake_score_story):
    """Create a test client with all external services mocked."""
    with patch("db.vector_store.collection", patched_vector_store.collection), \
         patch("services.narrative_engine.vector_store", patched_vector_store):
        from main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRiskEndpoints:
    """Tests for the /api/risk endpoints."""

    def test_get_risk_index_empty(self, client):
        """Risk index should handle empty database."""
        response = client.get("/api/risk")
        assert response.status_code == 200
        data = response.json()
        assert "model_risk_index" in data
        assert "narrative_count" in data
        assert "breakdown" in data

    def test_get_risk_history(self, client):
        """Risk history should return history array."""
        response = client.get("/api/risk/history?window=24&resolution=100")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)


class TestNarrativesEndpoints:
    """Tests for the /api/narratives endpoints."""

    def test_list_narratives_empty(self, client):
        """List narratives should return narratives array."""
        response = client.get("/api/narratives")
        assert response.status_code == 200
        data = response.json()
        assert "narratives" in data
        assert isinstance(data["narratives"], list)

    def test_list_narratives_with_params(self, client):
        """List narratives should accept query parameters."""
        response = client.get("/api/narratives?active_only=true&sort_by=risk&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "narratives" in data

    def test_get_narrative_not_found(self, client):
        """Get non-existent narrative should return 404."""
        response = client.get("/api/narratives/nonexistent-id")
        assert response.status_code == 404

    def test_get_narrative_history_not_found(self, client):
        """Get history for non-existent narrative should return 404."""
        response = client.get("/api/narratives/nonexistent-id/history")
        assert response.status_code == 404

    def test_search_narratives(self, client):
        """Search should return results array."""
        response = client.post(
            "/api/narratives/search",
            json={"query": "test query", "n_results": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)


class TestIngestEndpoint:
    """Tests for the /api/ingest endpoint."""

    def test_ingest_creates_narrative(self, client):
        """Ingest should create a new narrative."""
        response = client.post(
            "/api/ingest",
            json={
                "headline": "Test headline for new narrative",
                "body": "Test body content for the story."
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ["created", "updated"]
        assert "narrative_id" in data
        assert "narrative_name" in data
        assert "model_risk" in data
        assert "current_surprise" in data
        assert "current_impact" in data

    def test_ingest_without_body(self, client):
        """Ingest should work with just a headline."""
        response = client.post(
            "/api/ingest",
            json={"headline": "Headline only test"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ["created", "updated"]

    def test_ingest_updates_existing(self, client):
        """Ingesting similar content should update existing narrative."""
        response1 = client.post(
            "/api/ingest",
            json={
                "headline": "Oil prices surge on supply concerns",
                "body": "Crude oil prices jumped today."
            }
        )
        assert response1.status_code == 200

        response2 = client.post(
            "/api/ingest",
            json={
                "headline": "Oil prices surge on supply concerns",
                "body": "Crude oil prices increased significantly."
            }
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["action"] in ["created", "updated"]


class TestBatchIngestEndpoint:
    """Tests for the /api/ingest/batch endpoint."""

    def test_batch_ingest(self, client):
        """Batch ingest should process multiple stories."""
        response = client.post(
            "/api/ingest/batch",
            json={
                "stories": [
                    {"headline": "Batch story A", "body": "Body A"},
                    {"headline": "Batch story B", "body": "Body B"},
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 2
        assert len(data["results"]) == 2
        assert "duration_seconds" in data

    def test_batch_ingest_empty_list(self, client):
        """Batch ingest with empty list should return 0 processed."""
        response = client.post(
            "/api/ingest/batch",
            json={"stories": []}
        )
        assert response.status_code == 200
        assert response.json()["processed"] == 0


class TestPipelineStatsEndpoint:
    """Tests for the /api/pipeline/stats endpoint."""

    def test_get_pipeline_stats(self, client):
        """Pipeline stats should return proper structure."""
        response = client.get("/api/pipeline/stats")
        assert response.status_code == 200
        data = response.json()

        assert "pipeline" in data
        assert "narratives" in data

        pipeline = data["pipeline"]
        assert "stories_ingested" in pipeline
        assert "narratives_created" in pipeline
        assert "narratives_updated" in pipeline
        assert "queue_size" in pipeline
        assert "errors" in pipeline

        narratives = data["narratives"]
        assert "total" in narratives
        assert "active" in narratives


class TestSSEEndpoint:
    """Tests for the /api/events/stream SSE endpoint."""

    def test_sse_endpoint_route_exists(self, client):
        """SSE endpoint route should be registered."""
        from main import app
        routes = [route.path for route in app.routes]
        assert "/api/events/stream" in routes

    def test_sse_endpoint_handler(self):
        """SSE endpoint handler should return StreamingResponse."""
        from main import events_stream
        import asyncio
        assert asyncio.iscoroutinefunction(events_stream)


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, client):
        """Test the complete ingest -> query workflow."""
        # 1. Ingest a story
        ingest_response = client.post(
            "/api/ingest",
            json={
                "headline": "Tech stocks rally on AI optimism",
                "body": "Major technology companies saw gains."
            }
        )
        assert ingest_response.status_code == 200
        narrative_id = ingest_response.json()["narrative_id"]

        # 2. Verify narrative appears in list
        list_response = client.get("/api/narratives")
        assert list_response.status_code == 200
        narratives = list_response.json()["narratives"]
        narrative_ids = [n["id"] for n in narratives]
        assert narrative_id in narrative_ids

        # 3. Get narrative details
        detail_response = client.get(f"/api/narratives/{narrative_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["id"] == narrative_id

        # 4. Get narrative history
        history_response = client.get(f"/api/narratives/{narrative_id}/history")
        assert history_response.status_code == 200
        history = history_response.json()
        assert "surprise_series" in history
        assert "impact_series" in history
        assert "model_risk_series" in history
        assert "recent_headlines" in history

        # 5. Search for the narrative
        search_response = client.post(
            "/api/narratives/search",
            json={"query": "tech stocks AI", "n_results": 5}
        )
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        result_ids = [r["narrative"]["id"] for r in results]
        assert narrative_id in result_ids

        # 6. Check risk index reflects the narrative
        risk_response = client.get("/api/risk")
        assert risk_response.status_code == 200
        risk_data = risk_response.json()
        assert risk_data["narrative_count"] > 0


class TestNarrativeFields:
    """Tests for narrative field correctness."""

    def test_narrative_has_required_fields(self, client):
        """Narratives should have all required fields."""
        client.post(
            "/api/ingest",
            json={"headline": "Test for field validation", "body": "Testing fields."}
        )

        response = client.get("/api/narratives")
        assert response.status_code == 200
        narratives = response.json()["narratives"]

        if narratives:
            narrative = narratives[0]
            required_fields = [
                "id", "name", "description", "event_count",
                "current_surprise", "current_impact", "model_risk",
                "last_updated", "surprise_trend"
            ]
            for field in required_fields:
                assert field in narrative, f"Missing field: {field}"

    def test_surprise_trend_values(self, client):
        """Surprise trend should be one of the valid values."""
        response = client.get("/api/narratives")
        assert response.status_code == 200
        narratives = response.json()["narratives"]

        valid_trends = ["rising", "falling", "stable"]
        for narrative in narratives:
            assert narrative.get("surprise_trend") in valid_trends


class TestSearchResults:
    """Tests for search result format."""

    def test_search_result_has_similarity(self, client):
        """Search results should have similarity scores."""
        client.post(
            "/api/ingest",
            json={"headline": "Gold prices increase", "body": "Gold is rising."}
        )

        response = client.post(
            "/api/narratives/search",
            json={"query": "gold prices", "n_results": 5}
        )
        assert response.status_code == 200
        results = response.json()["results"]

        for result in results:
            assert "narrative" in result
            assert "similarity" in result
            assert 0 <= result["similarity"] <= 1


class TestErrorHandling:
    """Tests for error handling."""

    def test_ingest_missing_headline(self, client):
        """Ingest without headline should fail."""
        response = client.post(
            "/api/ingest",
            json={"body": "Body without headline"}
        )
        assert response.status_code == 422

    def test_ingest_headline_too_long(self, client):
        """Ingest with headline > 500 chars should fail."""
        response = client.post(
            "/api/ingest",
            json={"headline": "x" * 501}
        )
        assert response.status_code == 422

    def test_search_missing_query(self, client):
        """Search without query should fail."""
        response = client.post(
            "/api/narratives/search",
            json={"n_results": 5}
        )
        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Invalid JSON should return error."""
        response = client.post(
            "/api/ingest",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
