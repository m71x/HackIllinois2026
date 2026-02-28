from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import ingest, narratives, risk

app = FastAPI(title="Real-World Model Risk Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router,     prefix="/api/ingest",     tags=["ingest"])
app.include_router(narratives.router, prefix="/api/narratives", tags=["narratives"])
app.include_router(risk.router,       prefix="/api/risk",       tags=["risk"])


@app.get("/health")
def health():
    return {"status": "ok"}
