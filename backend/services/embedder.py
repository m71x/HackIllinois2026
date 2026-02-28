"""
Text embedding service — backed by the Modal-deployed Embedder class.

The Modal app must be deployed before this will work:
    modal deploy model/modal_app.py

Public interface is identical to the old local version so nothing else changes:
    embed_text(text: str)          -> list[float]
    embed_batch(texts: list[str])  -> list[list[float]]
"""

import modal
from core.config import settings

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = modal.Cls.lookup(settings.modal_app_name, "Embedder")
    return _embedder


def embed_text(text: str) -> list[float]:
    return _get_embedder()().embed.remote(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _get_embedder()().embed_batch.remote(texts)
