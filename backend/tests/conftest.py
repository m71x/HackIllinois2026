"""
Shared test fixtures and patches.

Key concerns
------------
- Modal is not installed in CI/test environments; injected as a fake sys module.
- ChromaDB uses an EphemeralClient per test so nothing persists to disk.
- yfinance is network-bound; patched per test in ticker tests.
"""

import sys
import math
import types
import chromadb
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Inject a fake 'modal' module before any backend code imports it.
# This must happen at collection time (module level), not inside a fixture,
# because embedder.py and llm_client.py import modal at the top of the file.
# ---------------------------------------------------------------------------

def _make_fake_modal():
    mod = types.ModuleType("modal")
    mod.Cls = MagicMock()
    mod.App = MagicMock()
    mod.Secret = MagicMock()
    mod.Image = MagicMock()
    mod.enter = MagicMock()
    mod.method = MagicMock()
    return mod

if "modal" not in sys.modules:
    sys.modules["modal"] = _make_fake_modal()


# ---------------------------------------------------------------------------
# In-memory ChromaDB collection — fresh per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def ephemeral_collection():
    import uuid
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        name=f"test_{uuid.uuid4().hex}",   # unique name prevents cross-test leakage
        metadata={"hnsw:space": "cosine"},
    )
    return col


@pytest.fixture()
def patched_vector_store(ephemeral_collection):
    """
    Swap the module-level ChromaDB collection for an ephemeral one.
    Restores the original after each test.
    """
    import db.vector_store as vs
    original = vs.collection
    vs.collection = ephemeral_collection
    yield vs
    vs.collection = original


# ---------------------------------------------------------------------------
# Deterministic unit vectors for testing
# ---------------------------------------------------------------------------

def make_unit_vector(seed: float, dim: int = 384) -> list[float]:
    v = [math.sin(seed + i * 0.1) for i in range(dim)]
    mag = sum(x ** 2 for x in v) ** 0.5
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# Fake embedder patches
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_embed_text():
    # Patch where it is used (imported into narrative_engine), not where defined
    with patch("services.narrative_engine.embed_text",
               side_effect=lambda text: make_unit_vector(hash(text) % 100)) as m1, \
         patch("services.embedder.embed_text",
               side_effect=lambda text: make_unit_vector(hash(text) % 100)) as m2:
        yield m1


@pytest.fixture()
def fake_embed_batch():
    with patch("services.embedder.embed_batch",
               side_effect=lambda texts: [make_unit_vector(hash(t) % 100) for t in texts]) as m1, \
         patch("api.routes.tickers.embed_batch",
               side_effect=lambda texts: [make_unit_vector(hash(t) % 100) for t in texts]) as m2:
        yield m1


# ---------------------------------------------------------------------------
# Fake LLM patches — patch at point-of-use (narrative_engine imports these directly)
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_label_narrative():
    with patch("services.narrative_engine.label_narrative",
               return_value={"name": "Fed tightening cycle",
                             "description": "Rising interest rates driven by Fed policy"}) as m:
        yield m


@pytest.fixture()
def fake_score_story():
    with patch("services.narrative_engine.score_story",
               return_value={"surprise": 0.6, "impact": 0.7}) as m:
        yield m
