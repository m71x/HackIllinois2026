from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    # Modal (GPU embeddings)
    modal_app_name: str = "model-risk-llm"   # must match APP_NAME in modal_app.py

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "narratives"

    # Narrative routing threshold (cosine distance [0, 2])
    # Stories with best_distance >= threshold spawn a new narrative direction.
    new_narrative_threshold: float = 0.40

    # Pipeline — background auto-scrape
    auto_start_pipeline: bool = True          # start pipeline on boot
    pipeline_num_workers: int = 32            # concurrent routing workers
    poll_interval_seconds: int = 120          # scrape every 2 minutes
    pipeline_lookback_minutes: int = 30       # pull stories from last 30 min per poll
    pipeline_max_per_source: int = 100        # max items per RSS feed per poll
    pipeline_sources: list[str] = ["rss", "newsapi", "twitter"]

    # Startup bulk ingest — pull historical RSS stories once on boot
    bulk_ingest_on_startup: bool = True
    bulk_ingest_lookback_hours: int = 168     # reach back 7 days of RSS history
    bulk_ingest_max_per_source: int = 500     # max items per feed for the bulk pull

    # NewsAPI  (newsapi.org — free tier: 100 req/day, 1-month lookback)
    newsapi_key: Optional[str] = None

    # Twitter/X API v2  (developer.twitter.com — bearer token, app-only auth)
    twitter_bearer_token: Optional[str] = None


settings = Settings()
