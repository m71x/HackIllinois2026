"""
Text embedding service — backed by the Modal-deployed Embedder class.

The Modal app must be deployed before this will work:
    modal deploy model/modal_app.py

Public interface:
    embed_text(text: str)          -> list[float]   # 384-dim L2-normalized
    embed_batch(texts: list[str])  -> list[list[float]]
"""

import modal
from core.config import settings

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            _embedder = modal.Cls.lookup(settings.modal_app_name, "Embedder")
        except Exception:
            _embedder = None
    return _embedder


def embed_text(text: str) -> list[float]:
    embedder = _get_embedder()
    if embedder is None:
        raise RuntimeError(
            "Modal Embedder not available. Deploy with: modal deploy model/modal_app.py"
        )
    return embedder().embed.remote(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    if embedder is None:
        raise RuntimeError(
            "Modal Embedder not available. Deploy with: modal deploy model/modal_app.py"
        )
    return embedder().embed_batch.remote(texts)
