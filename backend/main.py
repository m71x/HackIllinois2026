from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from api.routes import ingest, narratives, risk, pipeline, tickers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    if settings.auto_start_pipeline:
        from services.pipeline import start_pipeline
        await start_pipeline()

    yield

    # ---- shutdown ----
    from services.pipeline import stop_pipeline
    await stop_pipeline()


app = FastAPI(title="Real-World Model Risk Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router,    prefix="/api/ingest",    tags=["ingest"])
app.include_router(narratives.router, prefix="/api/narratives", tags=["narratives"])
app.include_router(risk.router,      prefix="/api/risk",      tags=["risk"])
app.include_router(pipeline.router,  prefix="/api/pipeline",  tags=["pipeline"])
app.include_router(tickers.router,   prefix="/api/tickers",   tags=["tickers"])


@app.get("/health")
def health():
    return {"status": "ok"}
