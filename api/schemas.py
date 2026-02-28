"""
Shared Request & Response Schemas
==================================
Source of truth for every request body and response body in the REST API.

Backend team:  implement these shapes in backend/api/routes/*.py
Frontend team: reference rest_api.md (same shapes, described in JSON)
Model team:    no direct dependency — see model_contract.py instead

All score values (surprise, impact, model_risk) are floats in [0.0, 1.0].
All timestamps are Unix epoch seconds (float).
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# Shared sub-objects
# ---------------------------------------------------------------------------

class TimeSeriesPoint(BaseModel):
    timestamp: float
    value: float


class NarrativeSummary(BaseModel):
    """Returned in list endpoints and search results. No time series."""
    id: str
    name: str
    description: str
    event_count: int
    current_surprise: Optional[float]
    current_impact: Optional[float]
    model_risk: Optional[float]
    last_updated: float
    is_active: bool
    surprise_trend: Optional[Literal["rising", "falling", "stable"]]
    impact_trend: Optional[Literal["rising", "falling", "stable"]]


class NarrativeDetail(NarrativeSummary):
    """Returned by GET /api/narratives/{id}. Includes full time series."""
    created_at: float
    surprise_series: list[TimeSeriesPoint]
    impact_series: list[TimeSeriesPoint]
    recent_headlines: list[str]


class NarrativeHistory(BaseModel):
    """Returned by GET /api/narratives/{id}/history. Chart-ready."""
    id: str
    name: str
    description: str
    surprise_series: list[TimeSeriesPoint]
    impact_series: list[TimeSeriesPoint]
    model_risk_series: list[TimeSeriesPoint]   # sqrt(surprise × impact) at each point
    event_count: int
    recent_headlines: list[str]


# ---------------------------------------------------------------------------
# Scrape-and-ingest
# ---------------------------------------------------------------------------

class ScrapeRequest(BaseModel):
    lookback_minutes: int = Field(default=60, ge=1, le=10080)
    max_per_source: int = Field(default=50, ge=1, le=100)
    sources: list[str] = Field(default=["newsapi", "twitter"])
    news_query: Optional[str] = None       # None = use default financial query
    twitter_query: Optional[str] = None    # None = use default financial query
    dry_run: bool = False


class NarrativeTouched(BaseModel):
    id: str
    name: str
    action: Literal["created", "updated"]
    model_risk: Optional[float]
    event_count: int


class StoryPreview(BaseModel):
    headline: str
    source: str
    body: str


class ScrapeRunResult(BaseModel):
    fetched: int                           # new stories after dedup
    duplicates_skipped: int                # already seen this session
    ingested: int                          # successfully processed
    narratives_created: int
    narratives_updated: int
    errors: int
    duration_seconds: float
    dedup_cache_size: int                  # total session cache entries
    per_source: dict                       # {"newsapi": int, "twitter": int}
    narratives_touched: list[NarrativeTouched]
    dry_run: bool
    stories_preview: Optional[list[StoryPreview]] = None  # only on dry_run


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    headline: str = Field(..., min_length=1, max_length=500)
    body: str = Field(default="", max_length=5000)
    source: str = Field(default="manual")


class IngestResponse(BaseModel):
    action: Literal["created", "updated"]
    narrative_id: str
    narrative_name: str
    best_distance: Optional[float]    # null if DB was empty on first ingest
    threshold: float
    current_surprise: Optional[float]
    current_impact: Optional[float]
    model_risk: Optional[float]
    narrative_event_count: int


class BatchIngestRequest(BaseModel):
    stories: list[IngestRequest]
    max_stories: int = Field(default=50, le=100)


class BatchIngestResponse(BaseModel):
    processed: int
    results: list[IngestResponse]
    errors: list[str]
    duration_seconds: float


# ---------------------------------------------------------------------------
# Narratives
# ---------------------------------------------------------------------------

class NarrativeListResponse(BaseModel):
    narratives: list[NarrativeSummary]
    total: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    n_results: int = Field(default=5, ge=1, le=20)
    active_only: bool = False


class SearchResultItem(BaseModel):
    narrative: NarrativeSummary
    distance: float      # cosine distance [0, 2]; lower = more similar
    similarity: float    # 1 - (distance / 2); easier to display as a percentage


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total: int


class DeleteResponse(BaseModel):
    deleted: bool
    id: str


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class RiskBreakdownItem(BaseModel):
    id: str
    name: str
    surprise: Optional[float]
    impact: Optional[float]
    model_risk: Optional[float]
    event_count: int
    is_active: bool
    last_updated: float


class RiskResponse(BaseModel):
    model_risk_index: Optional[float]    # null if no narratives exist
    narrative_count: int
    active_narrative_count: int
    breakdown: list[RiskBreakdownItem]   # sorted by model_risk descending
    aggregation_method: Literal["max", "mean", "weighted"]
    computed_at: float


class RiskHistoryPoint(BaseModel):
    timestamp: float
    model_risk_index: Optional[float]    # null for empty time buckets


class RiskHistoryResponse(BaseModel):
    history: list[RiskHistoryPoint]
    window_hours: int
    resolution: int
    start_time: float
    end_time: float


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    n_context_narratives: int = Field(default=5, ge=1, le=10)


class ChatContextItem(BaseModel):
    id: str
    name: str
    description: str
    current_surprise: Optional[float]
    current_impact: Optional[float]
    model_risk: Optional[float]
    recent_headlines: list[str]
    event_count: int
    distance: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    context_narratives: list[ChatContextItem]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineStats(BaseModel):
    stories_ingested: int
    narratives_created: int
    narratives_updated: int
    errors: int
    started_at: Optional[float]
    last_ingested_at: Optional[float]
    queue_size: int


class PipelineStatsResponse(BaseModel):
    pipeline: PipelineStats
    narratives: dict   # {"total": int, "active": int}
    events: dict       # {"total_events_ingested": int}
    timestamp: float


# ---------------------------------------------------------------------------
# SSE event payload (for reference — not a REST response)
# ---------------------------------------------------------------------------

class SSEIngestPayload(BaseModel):
    """
    Shape of the 'result' field inside an SSE 'ingest' event.
    Full event: {"type": "ingest", "timestamp": float, "result": SSEIngestPayload}
    """
    action: Literal["created", "updated"]
    narrative_id: str
    narrative_name: str
    best_distance: Optional[float]
    threshold: float
    current_surprise: Optional[float]
    current_impact: Optional[float]
    model_risk: Optional[float]
    narrative_event_count: int
