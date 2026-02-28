"""
Modal deployment for LLM inference and text embeddings.

Deploy once with:
    modal deploy model/modal_app.py

Two classes are deployed:
    LLM      — Llama 3.3 70B on A10G, called via modal.Cls.lookup("model-risk-llm", "LLM")
    Embedder — all-MiniLM-L6-v2 on T4,  called via modal.Cls.lookup("model-risk-llm", "Embedder")
"""

import modal

LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, no HF token needed
APP_NAME = "model-risk-llm"

llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.6", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers")
)

app = modal.App(APP_NAME)

# HuggingFace secret needed to download gated Llama models.
# Create it with: modal secret create huggingface-secret HF_TOKEN=your_token
hf_secret = modal.Secret.from_name("huggingface-secret")


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
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=MODEL_NAME,
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
        messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
        Returns the assistant's response as a plain string.
        """
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

        # vLLM applies the model's chat template automatically
        outputs = self.llm.chat(messages, sampling_params=params)
        return outputs[0].outputs[0].text.strip()


@app.cls(
    gpu="T4",                    # T4 is sufficient and cheaper for embedding
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
        """Embed a single string. Returns a normalized 384-dim float list."""
        return self._model.encode(text, normalize_embeddings=True).tolist()

    @modal.method()
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one forward pass."""
        return self._model.encode(texts, normalize_embeddings=True).tolist()


# --- Local entrypoints for quick testing ---
@app.local_entrypoint()
def main():
    llm = LLM()
    response = llm.chat.remote(
        messages=[{"role": "user", "content": "What is the Federal Reserve?"}],
        max_tokens=128,
    )
    print("LLM:", response)

    embedder = Embedder()
    vec = embedder.embed.remote("energy supply shock")
    print("Embedding dim:", len(vec), "| first 4 values:", vec[:4])
