"""
Text embedding service — backed by the Modal-deployed Embedder class.

Falls back to local sentence-transformers if Modal is unavailable.

Batching design
---------------
Individual embed_text() calls are coalesced into batches before they hit
Modal, so N concurrent single-story calls cost one network round-trip
instead of N.  A batch is dispatched when either:

  - BATCH_MAX_SIZE items have accumulated (flush immediately), or
  - BATCH_FLUSH_INTERVAL seconds elapse since the first queued item.

embed_batch() bypasses the queue entirely and goes straight to Modal —
prefer it when the caller already holds a list of texts.

Modal setup
-----------
    pip install modal
    modal setup          # authenticate once
    modal deploy model/modal_app.py

Public API
----------
    embed_text(text: str)          -> list[float]       # 384-dim L2-normalised
    embed_batch(texts: list[str])  -> list[list[float]]
"""

import hashlib
import logging
import threading
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)

# ── Batching knobs ────────────────────────────────────────────────────────────
BATCH_MAX_SIZE       = 32    # dispatch when the queue reaches this size
BATCH_FLUSH_INTERVAL = 0.05  # seconds: max latency before an auto-flush

# ── Module-level singletons ───────────────────────────────────────────────────
_modal_cls   = None   # cached modal.Cls.lookup result (thread-safe read once set)
_local_model = None   # SentenceTransformer instance or the sentinel "mock"
_use_local   = False  # set to True permanently if Modal is unreachable

# ── Thread-safe batch queue ───────────────────────────────────────────────────
# Each slot: (text, threading.Event, result_holder)
# result_holder is a one-element list so the flusher can write the vector
# back to the waiting thread without sharing a mutable reference.
_pending: list[tuple[str, threading.Event, list]] = []
_queue_lock  = threading.Lock()
_flush_timer: Optional[threading.Timer] = None


# ── Modal helpers ─────────────────────────────────────────────────────────────

def _get_modal_cls():
    """Return the cached Modal Embedder class handle, or None on failure."""
    global _modal_cls, _use_local
    if _modal_cls is None and not _use_local:
        try:
            import modal
            _modal_cls = modal.Cls.lookup(settings.modal_app_name, "Embedder")  # type: ignore[attr-defined]
            logger.info("[embedder] Connected to Modal Embedder (%s)", settings.modal_app_name)
        except Exception as exc:
            _use_local = True
            logger.warning("[embedder] Modal unavailable (%s), switching to local fallback", exc)
    return _modal_cls


# ── Local / mock helpers ──────────────────────────────────────────────────────

def _get_local_model():
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[embedder] Loaded local sentence-transformers model")
        except ImportError:
            _local_model = "mock"
            logger.warning("[embedder] sentence-transformers not installed, using mock embeddings")
    return _local_model


def _mock_embedding(text: str) -> list[float]:
    """Deterministic mock embedding derived from MD5 hash — for dev/test only."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [(int(h[i % 16], 16) / 15.0) - 0.5 for i in range(384)]


def _local_embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_local_model()
    if model == "mock":
        return [_mock_embedding(t) for t in texts]
    return model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore[call-arg]


# ── Batch flusher (runs on timer thread or caller thread) ─────────────────────

def _flush_pending(items: list[tuple[str, threading.Event, list]]) -> None:
    """
    Send *items* to Modal (or local fallback) as a single batch call,
    then wake each waiting thread by setting its Event.
    """
    if not items:
        return

    texts = [text for text, _, _ in items]

    try:
        cls = _get_modal_cls()
        if cls is not None:
            # TODO: replace with modal_cls().embed_batch.remote(texts) once credentials
            # are configured.  The call below is the correct Modal pattern:
            #   cls = modal.Cls.lookup("model-risk-llm", "Embedder")
            #   vecs = cls().embed_batch.remote(texts)
            vecs = cls().embed_batch.remote(texts)
        else:
            vecs = _local_embed_batch(texts)
    except Exception as exc:
        logger.error("[embedder] Modal batch call failed (%s), falling back to local", exc)
        vecs = _local_embed_batch(texts)

    for (_, event, holder), vec in zip(items, vecs):
        holder.append(vec)
        event.set()


def _trigger_flush() -> None:
    """Timer callback: drain _pending and dispatch the batch."""
    global _pending, _flush_timer
    with _queue_lock:
        items, _pending = _pending, []
        _flush_timer = None
    _flush_pending(items)


def _schedule_flush() -> None:
    """Arm the auto-flush timer if one isn't already running."""
    global _flush_timer
    if _flush_timer is not None and _flush_timer.is_alive():
        return
    _flush_timer = threading.Timer(BATCH_FLUSH_INTERVAL, _trigger_flush)
    _flush_timer.daemon = True
    _flush_timer.start()


# ── Public API ────────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Embed a single string.

    The call is queued and this thread blocks until the batch is flushed
    (either because BATCH_MAX_SIZE was reached or BATCH_FLUSH_INTERVAL
    elapsed).  Many concurrent embed_text() calls therefore share a single
    Modal round-trip instead of each making their own.

    Returns a 384-dim L2-normalised float list.
    """
    global _pending, _flush_timer

    event  = threading.Event()
    holder: list = []           # flusher writes the result vector here

    with _queue_lock:
        _pending.append((text, event, holder))

        if len(_pending) >= BATCH_MAX_SIZE:
            # Queue is full — dispatch immediately on this thread.
            if _flush_timer is not None:
                _flush_timer.cancel()
                _flush_timer = None
            items, _pending = _pending, []
        else:
            items = None
            _schedule_flush()

    if items is not None:
        # This thread is responsible for flushing; result lands in holder directly.
        _flush_pending(items)
    else:
        # Another thread (or the timer) will flush; wait for our slot.
        event.wait()

    return holder[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple strings in one Modal forward pass.

    Bypasses the batch queue — call this when you already have a list
    of texts to embed (e.g. the /api/pipeline/process bulk path).

    Returns one 384-dim L2-normalised vector per input, in input order.
    """
    cls = _get_modal_cls()
    if cls is not None:
        try:
            return cls().embed_batch.remote(texts)
        except Exception as exc:
            logger.error("[embedder] Modal embed_batch failed (%s), falling back to local", exc)
    return _local_embed_batch(texts)
