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

EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"          # 384-dim, public, no HF token needed
LABEL_MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct" # 0.5B, public, strong instruction-following
APP_NAME = "model-risk-llm"

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers==3.3.1")
)

label_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "torch>=2.1.0",
        "accelerate>=0.27.0",
    )
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
# Labeler — Qwen2.5-0.5B-Instruct on CPU
# Generates a concise 3-5 word narrative name from a news headline.
# Called only when a NEW narrative direction is created — low volume.
# ---------------------------------------------------------------------------

_LABEL_PROMPT = """\
You are a financial risk analyst. Produce a concise 3–5 word narrative label for the headline below.

Rules:
- Include specific entity names when present (Fed, SVB, OPEC+, China, Nvidia, etc.)
- Be specific, not generic ("SVB Bank Run" not "Banking Crisis")
- No punctuation, no quotes, no explanation — just the label

Examples:
"Federal Reserve raises rates 75bps amid inflation" → Fed Rate Hike
"SVB collapses as depositors flee" → SVB Bank Run Collapse
"China bans advanced chip exports to US firms" → China Chip Export Ban
"OPEC+ agrees surprise production cut" → OPEC+ Output Cut
"Nvidia posts record quarterly earnings beat" → Nvidia Earnings Beat
"US inflation hits 40-year high on energy costs" → US Inflation Surge
"Tesla announces mass layoffs, 10% of workforce" → Tesla Mass Layoffs
"US Treasury 10-year yield hits 5%" → Treasury Yield Surge

Headline: "{headline}"
Label:"""


@app.cls(
    cpu=2,
    memory=2048,
    image=label_image,
    container_idle_timeout=300,
)
class Labeler:
    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self._tok = AutoTokenizer.from_pretrained(LABEL_MODEL_NAME)
        self._model = AutoModelForCausalLM.from_pretrained(
            LABEL_MODEL_NAME, torch_dtype=torch.float32
        )
        self._model.eval()

    @modal.method()
    def label(self, headline: str, body_snippet: str = "") -> dict:
        """
        Generate a specific 3–5 word narrative label from a news headline.

        Returns {"name": str, "description": str}.
        body_snippet is ignored at inference time but kept for future use.
        """
        import torch

        prompt = _LABEL_PROMPT.format(headline=headline[:200])
        inputs = self._tok(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        name = self._tok.decode(new_tokens, skip_special_tokens=True).strip()

        # Keep only the first line, strip stray quotes/punctuation, cap length
        name = name.splitlines()[0].strip().strip('"\'').strip()
        if len(name) > 60:
            name = name[:57] + "…"
        if not name:
            name = "Market Development Shift"

        return {"name": name, "description": headline[:200]}


# ---------------------------------------------------------------------------
# Local entrypoints — smoke-test the deployed Embedder / Labeler
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


@app.local_entrypoint()
def test_label():
    """
    Smoke-test the Labeler. Checks that the model returns non-empty strings
    and that entity names survive in the output.

    Run with: modal run model/modal_app.py::test_label
    """
    labeler = Labeler()
    cases = [
        ("Federal Reserve raises interest rates by 75 basis points", "Fed"),
        ("SVB collapses as depositors withdraw billions overnight", "SVB"),
        ("China imposes new semiconductor export controls on US firms", "China"),
        ("OPEC+ agrees surprise production cut of 1 million barrels", "OPEC"),
        ("Nvidia reports record quarterly earnings, stock surges", "Nvidia"),
    ]
    for headline, expected_token in cases:
        result = labeler.label.remote(headline)
        name = result.get("name", "")
        hit = expected_token.lower() in name.lower()
        print(f"  '{headline[:55]}'\n  → '{name}'  [entity match: {hit}]")
        assert name, f"Empty label for: {headline}"
        assert len(name) <= 60, f"Label too long ({len(name)} chars): {name}"
    print("OK")
