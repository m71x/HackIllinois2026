"""
Modal deployment — text embeddings only.

Setup (one-time)
----------------
    pip install modal
    modal setup   # authenticate

Deploy
------
    modal deploy model/modal_app.py

After deployment the Embedder class is reachable from anywhere:
    modal.Cls.lookup("model-risk-llm", "Embedder")

Smoke-test the live deployment
-------------------------------
    modal run model/modal_app.py                   # runs main() below
    modal run model/modal_app.py::test_embed       # test single embed
    modal run model/modal_app.py::test_embed_batch # test batch embed

Contract
--------
See api/model_contract.py for the interface the backend expects.
"""

import modal

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, public, no HF token needed
APP_NAME = "model-risk-llm"

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers==3.3.1")
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Embedder — all-MiniLM-L6-v2 on T4
# ---------------------------------------------------------------------------

@app.cls(
    gpu="T4",
    image=embed_image,
    container_idle_timeout=300,
)
class Embedder:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(EMBED_MODEL_NAME)

    @modal.method()
    def embed(self, text: str) -> list[float]:
        """
        Embed a single string. Returns a 384-dim L2-normalized float list.
        Used for query embedding in semantic search and single-story ingests.
        """
        return self._model.encode(text, normalize_embeddings=True).tolist()

    @modal.method()
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings in one forward pass.
        Used by POST /api/pipeline/process to batch-embed all buffered stories
        in a single Modal round-trip before routing into ChromaDB.

        Returns one 384-dim L2-normalized vector per input, in input order.
        """
        return self._model.encode(texts, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Local entrypoints — smoke-test the deployed Embedder
# Run with: modal run model/modal_app.py::<function_name>
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Quick sanity check: one embed call."""
    embedder = Embedder()
    vec = embedder.embed.remote("energy supply shock")
    print(f"Embedding dim: {len(vec)} | first 4 values: {[round(v, 4) for v in vec[:4]]}")
    assert len(vec) == 384, f"Expected 384-dim, got {len(vec)}"
    print("OK")


@app.local_entrypoint()
def test_embed():
    """Test single-string embedding. Verifies 384-dim L2-normalized output."""
    embedder = Embedder()
    texts = [
        "Federal Reserve raises interest rates",
        "energy supply shock from OPEC production cut",
        "regional banking stress following SVB collapse",
    ]
    for text in texts:
        vec = embedder.embed.remote(text)
        assert len(vec) == 384, f"Expected 384-dim, got {len(vec)}"
        magnitude = sum(x ** 2 for x in vec) ** 0.5
        assert abs(magnitude - 1.0) < 1e-4, f"Vector not unit-normalized: magnitude={magnitude:.6f}"
        print(f"embed('{text[:50]}') → dim={len(vec)}, magnitude={magnitude:.6f}")
    print("OK")


@app.local_entrypoint()
def test_embed_batch():
    """
    Test batch embedding. Verifies output order is preserved and all vectors are 384-dim.
    This is the critical path for POST /api/pipeline/process.
    """
    texts = [
        "Fed signals rate hike pause",
        "Oil prices surge on OPEC cut",
        "China imposes new tech export controls",
        "SVB collapse triggers bank run fears",
        "Treasury yields hit 16-year high",
    ]
    embedder = Embedder()
    vecs = embedder.embed_batch.remote(texts)

    assert len(vecs) == len(texts), f"Expected {len(texts)} vectors, got {len(vecs)}"
    for i, (text, vec) in enumerate(zip(texts, vecs)):
        assert len(vec) == 384, f"Vector {i} has wrong dim: {len(vec)}"
        magnitude = sum(x ** 2 for x in vec) ** 0.5
        assert abs(magnitude - 1.0) < 1e-4, f"Vector {i} not normalized: magnitude={magnitude:.6f}"
        print(f"[{i}] embed('{text[:40]}') → dim={len(vec)}, magnitude={magnitude:.6f}")

    dot_related = sum(a * b for a, b in zip(vecs[0], vecs[4]))   # both Fed/rates topics
    dot_unrelated = sum(a * b for a, b in zip(vecs[0], vecs[2])) # rates vs China tech
    print(f"\nCosine(Fed pause, Treasury yields)={dot_related:.4f}  [should be higher]")
    print(f"Cosine(Fed pause, China tech export)={dot_unrelated:.4f} [should be lower]")
    print("OK")
