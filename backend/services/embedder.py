"""
Text embedding service — backed by the Modal-deployed Embedder class.

The Modal app must be deployed before this will work:
    modal deploy model/modal_app.py

Falls back to local sentence-transformers if Modal is unavailable.

Public interface is identical to the old local version so nothing else changes:
    embed_text(text: str)          -> list[float]
    embed_batch(texts: list[str])  -> list[list[float]]
"""

import hashlib
from core.config import settings

_embedder = None
_local_model = None
_use_local = False


def _get_embedder():
    global _embedder, _use_local
    if _embedder is None and not _use_local:
        try:
            import modal
            _embedder = modal.Cls.lookup(settings.modal_app_name, "Embedder")
        except Exception:
            _use_local = True
            print("[embedder] Modal unavailable, using local fallback")
    return _embedder


def _get_local_model():
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _local_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[embedder] Loaded local sentence-transformers model")
        except ImportError:
            _local_model = "mock"
            print("[embedder] sentence-transformers not installed, using mock embeddings")
    return _local_model


def _mock_embedding(text: str) -> list[float]:
    """Generate a deterministic mock embedding based on text hash."""
    h = hashlib.md5(text.encode()).hexdigest()
    # Generate 384 floats from hash (repeating as needed)
    values = []
    for i in range(384):
        byte_idx = i % 16
        values.append((int(h[byte_idx], 16) / 15.0) - 0.5)
    return values


def embed_text(text: str) -> list[float]:
    embedder = _get_embedder()
    if embedder is not None:
        return embedder().embed.remote(text)

    # Local fallback
    model = _get_local_model()
    if model == "mock":
        return _mock_embedding(text)
    return model.encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    if embedder is not None:
        return embedder().embed_batch.remote(texts)

    # Local fallback
    model = _get_local_model()
    if model == "mock":
        return [_mock_embedding(t) for t in texts]
    return model.encode(texts).tolist()
