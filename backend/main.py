import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.routes import ingest, narratives, risk
from db import vector_store
from core.config import settings
from core.state import pipeline_stats, sse_subscribers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup/shutdown events."""
    # ---- startup ----
    if settings.auto_start_pipeline:
        try:
            from services.pipeline import start_pipeline
            await start_pipeline()
        except ImportError:
            pass  # Pipeline service not available

    yield

    # ---- shutdown ----
    try:
        from services.pipeline import stop_pipeline
        await stop_pipeline()
    except ImportError:
        pass  # Pipeline service not available


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
