import asyncio
import logging
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from api.routes import ingest, narratives, risk
from db import vector_store
from core.config import settings
from core.state import pipeline_stats, sse_subscribers

_startup_log = logging.getLogger("nexus.startup")


async def _check_modal_status():
    """Probe Modal at startup and log a clear status line."""
    try:
        import modal
        cls = modal.Cls.from_name(settings.modal_app_name, "Embedder")
        loop = asyncio.get_event_loop()

        # hydrate() validates the deployment exists on Modal's servers.
        # It's a lightweight metadata fetch — no inference is run.
        if asyncio.iscoroutinefunction(cls.hydrate):
            await cls.hydrate()
        else:
            await loop.run_in_executor(None, cls.hydrate)

        # Pre-warm the embedder singleton so the first call skips the lookup
        from services import embedder as _emb
        _emb._modal_cls = cls
        _emb._use_local = False

        _startup_log.info(
            "┌─ Modal Embedder  ✓  CONNECTED\n"
            "│  App  : %s\n"
            "│  Class: Embedder  (GPU: T4)\n"
            "└─ All embeddings will run on Modal GPU",
            settings.modal_app_name,
        )
    except Exception as exc:
        _startup_log.warning(
            "┌─ Modal Embedder  ✗  OFFLINE\n"
            "│  Reason : %s\n"
            "└─ Falling back to local sentence-transformers (CPU)",
            exc,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup/shutdown events."""
    # ── 1. Modal connectivity check ──────────────────────────────────────────
    await _check_modal_status()

    # ── 2. Start background polling pipeline ─────────────────────────────────
    if settings.auto_start_pipeline:
        try:
            from services.pipeline import start_pipeline
            await start_pipeline()
        except ImportError:
            pass

    # ── 3. Startup bulk ingest (background task — doesn't block HTTP) ────────
    if settings.bulk_ingest_on_startup:
        try:
            from services.pipeline import bulk_ingest
            asyncio.create_task(bulk_ingest())
            _startup_log.info(
                "Bulk ingest task queued — %dh lookback, %d max/feed, RSS only",
                settings.bulk_ingest_lookback_hours,
                settings.bulk_ingest_max_per_source,
            )
        except ImportError:
            pass

    yield

    # ── shutdown ─────────────────────────────────────────────────────────────
    try:
        from services.pipeline import stop_pipeline
        await stop_pipeline()
    except ImportError:
        pass


app = FastAPI(title="Real-World Model Risk Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core routes
app.include_router(ingest.router,     prefix="/api/ingest",     tags=["ingest"])
app.include_router(narratives.router, prefix="/api/narratives", tags=["narratives"])
app.include_router(risk.router,       prefix="/api/risk",       tags=["risk"])

# Optional routes - only include if modules exist
try:
    from api.routes import pipeline
    app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
except ImportError:
    pass

try:
    from api.routes import tickers
    app.include_router(tickers.router, prefix="/api/tickers", tags=["tickers"])
except ImportError:
    pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/modal/status")
def modal_status():
    """Report whether the Modal GPU embedder is connected or falling back to local CPU."""
    from services import embedder as _emb
    connected = (not _emb._use_local) and (_emb._modal_cls is not None)
    return {
        "connected": connected,
        "backend": "modal_gpu" if connected else "local_cpu",
        "modal_app": settings.modal_app_name if connected else None,
    }


@app.get("/api/pipeline/stats")
def get_pipeline_stats():
    """Get current pipeline statistics."""
    all_narratives = vector_store.get_all_narratives()
    active_count = len([n for n in all_narratives if n.model_risk and n.model_risk > 0.1])

    return {
        "pipeline": pipeline_stats,
        "narratives": {
            "total": len(all_narratives),
            "active": active_count,
        },
    }


async def event_generator():
    """Generate SSE events for connected clients."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    sse_subscribers.append(queue)

    try:
        # Send initial connected message
        yield f"data: {json.dumps({'type': 'connected'})}\n\n"

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    finally:
        sse_subscribers.remove(queue)


@app.get("/api/events/stream")
async def events_stream():
    """Server-Sent Events endpoint for real-time updates."""
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve the Frontend directory as static files (MUST be last — catch-all)
frontend_dir = Path(__file__).resolve().parent.parent / "Frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


