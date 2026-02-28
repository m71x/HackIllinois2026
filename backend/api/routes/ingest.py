from fastapi import APIRouter
from pydantic import BaseModel
from services.narrative_engine import ingest_story

router = APIRouter()


class IngestRequest(BaseModel):
    headline: str
    body: str


@router.post("")
def ingest(req: IngestRequest):
    return ingest_story(req.headline, req.body)
