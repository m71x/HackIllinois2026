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

    # Pipeline
    pipeline_num_workers: int = 2
    poll_interval_seconds: int = 60

    # RSS sources (override via env as JSON list)
    rss_sources: list[str] = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ]

    # Reddit (optional — leave empty to disable)
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_subreddits: list[str] = ["stocks", "investing", "worldnews"]


settings = Settings()
