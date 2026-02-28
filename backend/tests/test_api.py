"""
FastAPI route tests using TestClient with all external services mocked.
Modal, ChromaDB (ephemeral), and yfinance are all patched.
"""

import math
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def make_unit_vector(seed: int, dim: int = 384) -> list[float]:
    v = [math.sin(seed + i * 0.01) for i in range(dim)]
    mag = sum(x ** 2 for x in v) ** 0.5
    return [x / mag for x in v]


@pytest.fixture()
def client(patched_vector_store, fake_embed_text, fake_embed_batch,
           fake_label_narrative, fake_score_story):
    """TestClient with all external services patched and in-memory ChromaDB."""
    with patch("db.vector_store.collection", patched_vector_store.collection), \
         patch("services.narrative_engine.vector_store", patched_vector_store):
        from main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Narratives
# ---------------------------------------------------------------------------

class TestNarrativesRoutes:
    def test_list_empty(self, client):
        r = client.get("/api/narratives")
        assert r.status_code == 200
        assert r.json() == []

    def test_get_nonexistent_404(self, client):
        r = client.get("/api/narratives/does-not-exist")
        assert r.status_code == 404

    def test_list_after_ingest(self, client):
        # Ingest a story to create a narrative
        client.post("/api/ingest", json={"headline": "Fed raises rates", "body": "Details..."})
        r = client.get("/api/narratives")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["name"] == "Fed tightening cycle"


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class TestIngestRoutes:
    def test_single_ingest_creates_narrative(self, client):
        r = client.post("/api/ingest", json={"headline": "Inflation hits 8%", "body": "CPI data..."})
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "created"
        assert data["narrative_id"]
        assert data["narrative_name"]

    def test_ingest_missing_headline_422(self, client):
        r = client.post("/api/ingest", json={"body": "No headline here"})
        assert r.status_code == 422

    def test_ingest_headline_too_long_422(self, client):
        r = client.post("/api/ingest", json={"headline": "x" * 501})
        assert r.status_code == 422

    def test_batch_ingest(self, client):
        r = client.post("/api/ingest/batch", json={
            "stories": [
                {"headline": "Story A", "body": "Body A"},
                {"headline": "Story B", "body": "Body B"},
            ]
        })
        assert r.status_code == 200
        data = r.json()
        assert data["processed"] == 2
        assert len(data["results"]) == 2
        assert "duration_seconds" in data

    def test_batch_ingest_empty_list(self, client):
        r = client.post("/api/ingest/batch", json={"stories": []})
        assert r.status_code == 200
        assert r.json()["processed"] == 0


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class TestRiskRoutes:
    def test_risk_empty_db(self, client):
        from api.routes import risk as risk_module
        r = client.get("/api/risk")
        assert r.status_code == 200
        data = r.json()
        # With no narratives, index should be null
        assert data["model_risk_index"] is None or isinstance(data["model_risk_index"], float)

    def test_risk_after_ingest(self, client):
        client.post("/api/ingest", json={"headline": "Banks under stress", "body": "SVB..."})
        r = client.get("/api/risk")
        assert r.status_code == 200
        data = r.json()
        assert data["narrative_count"] == 1
        assert isinstance(data["model_risk_index"], float)
        assert 0.0 <= data["model_risk_index"] <= 1.0


# ---------------------------------------------------------------------------
# Pipeline buffer
# ---------------------------------------------------------------------------

class TestPipelineRoutes:
    def test_buffer_empty_on_start(self, client):
        r = client.get("/api/pipeline/buffer")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_scrape_with_buffer_flag_does_not_ingest(self, client):
        """buffer=true should add to buffer, NOT touch ChromaDB."""
        from services.story_buffer import buffer as story_buffer
        story_buffer.clear()

        with patch("services.scraper.scrape") as mock_scrape:
            from services.scraper import RawStory
            import time
            mock_scrape.return_value = [
                RawStory("Buffered headline", "body", "test", time.time(), "http://x.com"),
            ]
            with patch("services.scraper.scrape_newsapi", return_value=[]):
                with patch("services.scraper.scrape_twitter", return_value=[]):
                    r = client.post("/api/ingest/scrape", json={
                        "sources": ["newsapi"],
                        "buffer": True,
                    })

        assert r.status_code == 200
        data = r.json()
        assert data["buffer_mode"] is True
        assert data["ingested"] == 0

    def test_clear_buffer(self, client):
        from services.story_buffer import buffer as story_buffer
        from services.scraper import RawStory
        import time
        story_buffer.add(RawStory("H", "B", "src", "u", time.time()))
        assert story_buffer.size() == 1
        r = client.delete("/api/pipeline/buffer")
        assert r.status_code == 200
        assert r.json()["cleared"] == 1
        assert story_buffer.size() == 0

    def test_pipeline_stats(self, client):
        r = client.get("/api/pipeline/stats")
        assert r.status_code == 200
        data = r.json()
        assert "pipeline" in data
        assert "narratives" in data
        assert "events" in data


# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

class TestTickerRoutes:
    def _mock_ticker_info(self):
        from services.ticker_service import TickerInfo
        return TickerInfo(
            symbol="JPM",
            name="JPMorgan Chase & Co.",
            sector="Financial Services",
            industry="Banks—Diversified",
            description="Global banking firm.",
            market_cap=500_000_000_000,
            embed_text="JPMorgan Chase & Co. (JPM) — Financial Services, Banks—Diversified\nGlobal banking firm.",
        )

    def test_relate_returns_results(self, client):
        with patch("api.routes.tickers.get_ticker_info_batch") as mock_batch, \
             patch("api.routes.tickers.embed_batch") as mock_emb:
            mock_batch.return_value = {"JPM": self._mock_ticker_info()}
            mock_emb.return_value = [make_unit_vector(1)]

            r = client.post("/api/tickers/relate", json={"tickers": ["JPM"]})

        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["ticker"] == "JPM"

    def test_relate_invalid_ticker_goes_to_errors(self, client):
        with patch("api.routes.tickers.get_ticker_info_batch") as mock_batch, \
             patch("api.routes.tickers.embed_batch") as mock_emb:
            mock_batch.return_value = {"ZZZBAD": ValueError("No data found")}
            mock_emb.return_value = []

            r = client.post("/api/tickers/relate", json={"tickers": ["ZZZBAD"]})

        assert r.status_code == 200
        data = r.json()
        assert "ZZZBAD" in data["errors"]
        assert data["results"] == []

    def test_relate_empty_tickers_422(self, client):
        r = client.post("/api/tickers/relate", json={"tickers": []})
        assert r.status_code == 422

    def test_expose_narrative_not_found(self, client):
        r = client.post("/api/tickers/expose", json={
            "narrative_id": "nonexistent-id",
            "tickers": ["JPM"],
        })
        assert r.status_code == 404

    def test_expose_ranks_by_similarity(self, client, patched_vector_store, fake_label_narrative, fake_score_story):
        vs = patched_vector_store
        from models.narrative import NarrativeDirection
        n = NarrativeDirection(name="Banking stress", description="Regional bank failures")
        n.append_surprise(0.7)
        n.append_impact(0.8)
        n.add_headline("SVB collapses")
        narrative_emb = make_unit_vector(10)
        vs.add_narrative(n, embedding=narrative_emb)

        with patch("api.routes.tickers.get_ticker_info_batch") as mock_batch, \
             patch("api.routes.tickers.embed_batch") as mock_emb, \
             patch("api.routes.tickers.vector_store", vs):

            from services.ticker_service import TickerInfo
            mock_batch.return_value = {
                "JPM": TickerInfo("JPM", "JPMorgan", "Finance", "Banks", "Banking firm", None, "JPMorgan (JPM)..."),
                "XOM": TickerInfo("XOM", "Exxon", "Energy", "Oil", "Oil firm", None, "Exxon (XOM)..."),
            }
            # JPM is close to the narrative, XOM is far
            mock_emb.return_value = [make_unit_vector(10), make_unit_vector(200)]

            r = client.post("/api/tickers/expose", json={
                "narrative_id": n.id,
                "tickers": ["JPM", "XOM"],
            })

        assert r.status_code == 200
        data = r.json()
        assert len(data["rankings"]) == 2
        # JPM (seed=10, same as narrative seed=10) should rank first
        assert data["rankings"][0]["ticker"] == "JPM"
        assert data["rankings"][0]["similarity"] > data["rankings"][1]["similarity"]

    def test_cache_clear(self, client):
        r = client.delete("/api/tickers/cache")
        assert r.status_code == 200
        assert r.json()["cleared"] == "all"
