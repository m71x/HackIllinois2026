"""
Text embedding service.
Uses a local sentence-transformers model for fast, free embeddings.
Swap embed_text() implementation if you want to use an API-based embedder.
"""

from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, ~80MB, runs on CPU


def embed_text(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts, normalize_embeddings=True).tolist()
