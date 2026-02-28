"""
Modal deployment — LLM inference and text embeddings.

Setup (one-time)
----------------
    pip install modal
    modal setup                                              # authenticate
    modal secret create huggingface-secret HF_TOKEN=<token> # gated Llama access

Deploy
------
    modal deploy model/modal_app.py

After deployment both classes are reachable from anywhere:
    modal.Cls.lookup("model-risk-llm", "LLM")
    modal.Cls.lookup("model-risk-llm", "Embedder")

Smoke-test the live deployment
-------------------------------
    modal run model/modal_app.py                   # runs main() below
    modal run model/modal_app.py::test_label       # test label_narrative prompt
    modal run model/modal_app.py::test_score       # test score_story prompt
    modal run model/modal_app.py::test_embed       # test single embed
    modal run model/modal_app.py::test_embed_batch # test batch embed

Contract
--------
See api/model_contract.py for the interface the backend expects.
"""

import modal

LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, public, no HF token needed
APP_NAME = "model-risk-llm"

llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.6", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers==3.3.1")
)

app = modal.App(APP_NAME)

# HuggingFace secret needed to download gated Llama models.
# Create with: modal secret create huggingface-secret HF_TOKEN=your_token
hf_secret = modal.Secret.from_name("huggingface-secret")


# ---------------------------------------------------------------------------
# LLM — Llama 3.3 70B on A10G via vLLM
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A10G",
    image=llm_image,
    timeout=600,
    container_idle_timeout=300,
    secrets=[hf_secret],
)

class LLM:
    @modal.enter()
    def load_model(self):
        from vllm import LLM as VllmLLM, SamplingParams
        self.llm = VllmLLM(
            model=LLM_MODEL_NAME,
            max_model_len=4096,
            dtype="bfloat16",
        )
        self.SamplingParams = SamplingParams

    @modal.method()
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        """
        OpenAI-style chat completion.

        Parameters
        ----------
        messages    : list of {"role": "user"|"assistant"|"system", "content": str}
        max_tokens  : 64 for score_story, 256 for label_narrative, 512 for chat
        temperature : 0.1 for JSON outputs, 0.3 for prose outputs

        Returns
        -------
        str — assistant reply, stripped of leading/trailing whitespace.
        """
        params = self.SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs = self.llm.chat(messages, sampling_params=params)
        return outputs[0].outputs[0].text.strip()


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
# Local entrypoints — smoke-test the deployed classes
# Run with: modal run model/modal_app.py::<function_name>
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Quick sanity check: one LLM call + one embed call."""
    llm = LLM()
    response = llm.chat.remote(
        messages=[{"role": "user", "content": "What is the Federal Reserve? One sentence."}],
        max_tokens=128,
        temperature=0.1,
    )
    print(f"LLM response: {response}")

    embedder = Embedder()
    vec = embedder.embed.remote("energy supply shock")
    print(f"Embedding dim: {len(vec)} | first 4 values: {[round(v, 4) for v in vec[:4]]}")
    assert len(vec) == 384, f"Expected 384-dim, got {len(vec)}"
    print("OK")


@app.local_entrypoint()
def test_label():
    """
    Test the label_narrative prompt path.
    Simulates what backend/services/llm_client.py :: label_narrative() sends.
    Expected output: valid JSON with 'name' and 'description' keys.
    """
    prompt = (
        "You are a financial risk analyst identifying persistent real-world narrative directions.\n\n"
        "A \"narrative direction\" is a broad, ongoing real-world theme that can drive market model breakdown.\n"
        "Examples:\n"
        "  - \"Energy supply shock\" — ongoing constraints on energy availability affecting markets\n"
        "  - \"Regional banking stress\" — deteriorating confidence in mid-size banks\n\n"
        "Given the news story below, identify which narrative direction it belongs to.\n\n"
        "Respond with ONLY valid JSON. No commentary. No markdown fences.\n\n"
        "{\"name\": \"<3 to 6 word label>\", \"description\": \"<one sentence describing the persistent narrative direction>\"}\n\n"
        "News story:\n"
        "Federal Reserve officials signaled Wednesday that interest rate cuts are unlikely before mid-year "
        "as inflation remains above target. Chair Powell said the committee needs greater confidence that "
        "inflation is sustainably moving toward 2% before easing policy."
    )
    llm = LLM()
    result = llm.chat.remote(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
    )
    print(f"label_narrative output:\n{result}")
    import json
    parsed = json.loads(result)
    assert "name" in parsed and "description" in parsed, "Missing required keys"
    print(f"\nParsed name: {parsed['name']}")
    print(f"Parsed description: {parsed['description']}")
    print("OK")


@app.local_entrypoint()
def test_score():
    """
    Test the score_story prompt path.
    Simulates what backend/services/llm_client.py :: score_story() sends.
    Expected output: valid JSON with 'surprise' and 'impact' keys in [0, 1].
    """
    prompt = (
        "You are a quantitative financial risk analyst.\n\n"
        "Narrative direction: Federal Reserve monetary tightening cycle\n"
        "\nCurrent narrative state:\n"
        "  - Surprise so far: 0.45\n"
        "  - Impact so far:   0.60\n"
        "Does this story ESCALATE, CONTINUE, or DE-ESCALATE the narrative?\n\n"
        "Score the following news story on two dimensions:\n\n"
        "SURPRISE [0.0–1.0]: How unexpected or regime-breaking is this development within the narrative?\n"
        "  0.0 = expected continuation already priced in\n"
        "  0.5 = moderate escalation, partially surprising\n"
        "  1.0 = sudden shock, reversal, or unprecedented development\n\n"
        "IMPACT [0.0–1.0]: How economically significant is this event?\n"
        "  0.0 = negligible market relevance\n"
        "  0.5 = affects one major sector or mid-cap companies\n"
        "  1.0 = systemic, multi-sector, global significance\n\n"
        "Respond with ONLY valid JSON. No commentary.\n"
        "{\"surprise\": <float 0.0–1.0>, \"impact\": <float 0.0–1.0>}\n\n"
        "News story:\n"
        "The Federal Reserve raised rates by 50 basis points in a surprise emergency meeting, "
        "the first unscheduled hike since 2020, citing a sudden surge in core PCE inflation to 4.8%."
    )
    llm = LLM()
    result = llm.chat.remote(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.1,
    )
    print(f"score_story output:\n{result}")
    import json
    parsed = json.loads(result)
    assert "surprise" in parsed and "impact" in parsed, "Missing required keys"
    assert 0.0 <= parsed["surprise"] <= 1.0, f"surprise out of range: {parsed['surprise']}"
    assert 0.0 <= parsed["impact"] <= 1.0, f"impact out of range: {parsed['impact']}"
    print(f"\nSurprise: {parsed['surprise']}")
    print(f"Impact:   {parsed['impact']}")
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
        print(f"embed('{text[:50]}...') → dim={len(vec)}, magnitude={magnitude:.6f}")
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
        print(f"[{i}] embed('{text[:40]}...') → dim={len(vec)}, magnitude={magnitude:.6f}")

    # Verify cosine similarity between related vs unrelated stories
    import math
    dot_related = sum(a * b for a, b in zip(vecs[0], vecs[4]))   # both Fed/rates topics
    dot_unrelated = sum(a * b for a, b in zip(vecs[0], vecs[2])) # rates vs China tech
    print(f"\nCosine(Fed pause, Treasury yields)={dot_related:.4f}  [should be higher]")
    print(f"Cosine(Fed pause, China tech export)={dot_unrelated:.4f} [should be lower]")
    print("OK")
