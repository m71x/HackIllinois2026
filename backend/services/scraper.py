"""
News & Tweet Scraper
====================
Pulls real-time story summaries from NewsAPI and tweets from Twitter/X API.

All scraping is controlled by ScrapeParams — a single object that exposes
every time-recency and volume knob. Pass it to scrape() for a one-shot pull,
or let pipeline.py call it on a schedule.

Sources:
    newsapi  — newsapi.org (requires NEWSAPI_KEY)
    twitter  — Twitter v2 recent search (requires TWITTER_BEARER_TOKEN)

Deduplication:
    A module-level DeduplicatingCache (SHA-256 hash, 10k entries) persists
    across calls so the same story is never ingested twice in a session.
"""

from __future__ import annotations

import hashlib
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RawStory:
    headline: str
    body: str
    source: str
    url: str = ""
    published_at: float = field(default_factory=time.time)


@dataclass
class ScrapeParams:
    """
    All controls for a single scrape run.

    Time recency
    ------------
    lookback_minutes : int
        Only pull stories published within the last N minutes.
        Examples:
            30   → last 30 minutes (very fresh, low volume)
            60   → last hour       (default, good balance)
            360  → last 6 hours    (broader sweep)
            1440 → last 24 hours   (maximum lookback for free NewsAPI tier)

    Volume
    ------
    max_per_source : int
        Maximum number of items to fetch per source.
        NewsAPI caps at 100 per request on the free tier.
        Twitter caps at 100 per request on the basic tier.

    Sources
    -------
    sources : list[str]
        Which sources to pull from. Any subset of ["newsapi", "twitter"].

    Query overrides
    ---------------
    news_query : str
        Override the default NewsAPI keyword query.
    twitter_query : str
        Override the default Twitter search query.
        Must comply with Twitter v2 query syntax.
        See: https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query

    Other
    -----
    dry_run : bool
        If True, fetch stories but do NOT ingest them. Returns the list for inspection.
    """
    lookback_minutes: int = 60
    max_per_source: int = 50
    sources: list[str] = field(default_factory=lambda: ["newsapi", "twitter"])

    news_query: str = (
        "economy OR inflation OR recession OR \"federal reserve\" OR \"interest rates\" "
        "OR \"stock market\" OR GDP OR \"trade war\" OR sanctions OR geopolitical "
        "OR \"supply chain\" OR \"central bank\" OR \"banking crisis\" OR \"credit risk\""
    )

    twitter_query: str = (
        "(economy OR inflation OR recession OR \"federal reserve\" OR \"stock market\" "
        "OR sanctions OR geopolitical OR \"banking crisis\" OR \"supply chain\" "
        "OR \"credit risk\" OR \"rate hike\") -is:retweet lang:en"
    )

    dry_run: bool = False


@dataclass
class ScrapeResult:
    """Summary of what happened during a scrape run."""
    fetched: int = 0
    duplicates_skipped: int = 0
    ingested: int = 0
    narratives_created: int = 0
    narratives_updated: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    per_source: dict = field(default_factory=dict)
    narratives_touched: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deduplication cache — module-level singleton, persists across scrape calls
# ---------------------------------------------------------------------------

class DeduplicatingCache:
    """LRU cache of SHA-256 content hashes. Prevents re-ingesting the same story."""

    def __init__(self, maxsize: int = 10_000):
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._maxsize = maxsize

    def _key(self, headline: str, body: str) -> str:
        content = f"{headline.strip()}{body.strip()[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()

    def is_seen(self, headline: str, body: str) -> bool:
        k = self._key(headline, body)
        if k in self._cache:
            self._cache.move_to_end(k)
            return True
        return False

    def mark_seen(self, headline: str, body: str):
        k = self._key(headline, body)
        self._cache[k] = True
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def size(self) -> int:
        return len(self._cache)


# Shared across all scrape calls within a server session
_cache = DeduplicatingCache(maxsize=10_000)


# ---------------------------------------------------------------------------
# NewsAPI scraper
# ---------------------------------------------------------------------------

def scrape_newsapi(params: ScrapeParams) -> list[RawStory]:
    """
    Fetch article summaries from NewsAPI.

    Returns article title as headline and description as body.
    Filters to articles published within lookback_minutes.
    """
    try:
        from newsapi import NewsApiClient
    except ImportError:
        logger.warning("newsapi-python not installed. Run: pip install newsapi-python")
        return []

    from core.config import settings
    if not settings.newsapi_key:
        logger.warning("NEWSAPI_KEY not set — skipping NewsAPI")
        return []

    client = NewsApiClient(api_key=settings.newsapi_key)
    since = datetime.utcnow() - timedelta(minutes=params.lookback_minutes)

    try:
        response = client.get_everything(
            q=params.news_query,
            from_param=since.strftime("%Y-%m-%dT%H:%M:%S"),
            language="en",
            sort_by="publishedAt",
            page_size=min(params.max_per_source, 100),
        )
    except Exception as e:
        logger.error(f"NewsAPI request failed: {e}")
        return []

    stories = []
    for article in response.get("articles", []):
        headline = (article.get("title") or "").strip()
        body = (article.get("description") or article.get("content") or "").strip()

        # NewsAPI returns "[Removed]" for deleted/paywalled articles
        if not headline or headline == "[Removed]":
            continue

        # Parse published_at — ISO8601 with Z suffix
        published_at = time.time()
        raw_ts = article.get("publishedAt")
        if raw_ts:
            try:
                published_at = datetime.fromisoformat(
                    raw_ts.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                pass

        source_name = (article.get("source") or {}).get("name", "unknown")
        stories.append(RawStory(
            headline=headline,
            body=body,
            source=f"newsapi:{source_name}",
            url=article.get("url", ""),
            published_at=published_at,
        ))

    logger.info(f"NewsAPI: fetched {len(stories)} articles (lookback={params.lookback_minutes}m)")
    return stories


# ---------------------------------------------------------------------------
# Twitter scraper
# ---------------------------------------------------------------------------

def scrape_twitter(params: ScrapeParams) -> list[RawStory]:
    """
    Fetch recent tweets from Twitter/X API v2.

    Uses the recent search endpoint — covers the last 7 days max.
    Requires TWITTER_BEARER_TOKEN (App-only auth, no user login needed).
    """
    try:
        import tweepy
    except ImportError:
        logger.warning("tweepy not installed. Run: pip install tweepy")
        return []

    from core.config import settings
    if not settings.twitter_bearer_token:
        logger.warning("TWITTER_BEARER_TOKEN not set — skipping Twitter")
        return []

    client = tweepy.Client(
        bearer_token=settings.twitter_bearer_token,
        wait_on_rate_limit=False,
    )

    since = datetime.now(timezone.utc) - timedelta(minutes=params.lookback_minutes)
    # Twitter API requires at least 10 results and at most 100 per request
    max_results = max(10, min(params.max_per_source, 100))

    try:
        response = client.search_recent_tweets(
            query=params.twitter_query,
            max_results=max_results,
            start_time=since,
            tweet_fields=["created_at", "text", "author_id"],
        )
    except tweepy.errors.TweepyException as e:
        logger.error(f"Twitter API request failed: {e}")
        return []

    stories = []
    for tweet in response.data or []:
        text = (tweet.text or "").strip()
        if not text:
            continue

        published_at = time.time()
        if tweet.created_at:
            try:
                published_at = tweet.created_at.timestamp()
            except Exception:
                pass

        stories.append(RawStory(
            headline=text[:280],   # tweet text is the headline; no separate body
            body="",
            source="twitter",
            url=f"https://twitter.com/i/web/status/{tweet.id}",
            published_at=published_at,
        ))

    logger.info(f"Twitter: fetched {len(stories)} tweets (lookback={params.lookback_minutes}m)")
    return stories


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scrape(params: ScrapeParams) -> list[RawStory]:
    """
    Fetch stories from all enabled sources, deduplicate, and return.

    Stories that have been seen before (within this server session) are
    silently dropped. The caller decides whether to ingest the result.
    """
    raw: list[RawStory] = []

    if "newsapi" in params.sources:
        raw.extend(scrape_newsapi(params))
    if "twitter" in params.sources:
        raw.extend(scrape_twitter(params))

    # Deduplicate
    fresh = []
    for story in raw:
        if _cache.is_seen(story.headline, story.body):
            continue
        _cache.mark_seen(story.headline, story.body)
        fresh.append(story)

    logger.info(
        f"scrape() total={len(raw)} fresh={len(fresh)} "
        f"skipped={len(raw) - len(fresh)} cache_size={_cache.size()}"
    )
    return fresh


def cache_size() -> int:
    """Current number of entries in the dedup cache."""
    return _cache.size()
