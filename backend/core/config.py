from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Modal
    modal_app_name: str = "model-risk-llm"   # must match APP_NAME in modal_app.py

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "narratives"

    # Narrative routing threshold (cosine distance [0, 2])
    # Stories with best_distance >= threshold spawn a new narrative direction.
    new_narrative_threshold: float = 0.40

    # Pipeline — background auto-scrape
    auto_start_pipeline: bool = False         # set to true to scrape automatically on boot
    pipeline_num_workers: int = 2
    poll_interval_seconds: int = 300          # scrape every 5 minutes by default
    pipeline_lookback_minutes: int = 10       # only pull stories from last 10 min per poll
    pipeline_max_per_source: int = 30         # max items per source per poll
    pipeline_sources: list[str] = ["newsapi", "twitter"]

    # NewsAPI  (newsapi.org — free tier: 100 req/day, 1-month lookback)
    newsapi_key: Optional[str] = None

    # Twitter/X API v2  (developer.twitter.com — bearer token, app-only auth)
    twitter_bearer_token: Optional[str] = None


settings = Settings()
