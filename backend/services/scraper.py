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

    # RSS feed URLs — no API key needed, no rate limits.
    # Fetched concurrently so total scrape time ≈ slowest single feed.
    rss_feeds: list[str] = field(default_factory=lambda: [
        # ── CNBC (reliable, no paywall) ───────────────────────────────────────
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",   # Finance
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",   # Business
        "https://www.cnbc.com/id/15839135/device/rss/rss.html",   # Economy
        "https://www.cnbc.com/id/20910258/device/rss/rss.html",   # Markets
        "https://www.cnbc.com/id/10000115/device/rss/rss.html",   # Earnings
        "https://www.cnbc.com/id/15839069/device/rss/rss.html",   # Investing
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # US News
        "https://www.cnbc.com/id/10000108/device/rss/rss.html",   # World
        # ── BBC ──────────────────────────────────────────────────────────────
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        # ── Yahoo Finance ────────────────────────────────────────────────────
        "https://finance.yahoo.com/news/rssindex",
        # ── Investopedia ─────────────────────────────────────────────────────
        "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline",
        # ── Federal Reserve / Central Banks ──────────────────────────────────
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.ecb.europa.eu/rss/press.html",
        # ── Energy & Commodities ─────────────────────────────────────────────
        "https://oilprice.com/rss/main",
        "https://www.eia.gov/rss/news.xml",
        "https://www.kitco.com/rss/news/",
        # ── Crypto / Digital Assets ──────────────────────────────────────────
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://bitcoinmagazine.com/feed",
        # ── Tech / AI ────────────────────────────────────────────────────────
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        # ── Business / Finance ───────────────────────────────────────────────
        "https://fortune.com/feed",
        "https://www.axios.com/feeds/feed.rss",
        "https://www.thestreet.com/rss/",
        "https://www.benzinga.com/feeds/",
        "https://abcnews.go.com/abcnews/businessheadlines",
        "https://feeds.nbcnews.com/nbcnews/public/business",
        "https://www.dw.com/en/rss/business/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        # ── Reddit financial communities (high-volume, no auth) ───────────────
        "https://www.reddit.com/r/investing/.rss",
        "https://www.reddit.com/r/stocks/.rss",
        "https://www.reddit.com/r/Economics/.rss",
        "https://www.reddit.com/r/wallstreetbets/.rss",
        "https://www.reddit.com/r/finance/.rss",
        "https://www.reddit.com/r/StockMarket/.rss",
        "https://www.reddit.com/r/options/.rss",
        "https://www.reddit.com/r/SecurityAnalysis/.rss",
        "https://www.reddit.com/r/MacroEconomics/.rss",
        "https://www.reddit.com/r/CryptoCurrency/.rss",
        "https://www.reddit.com/r/financialindependence/.rss",
        "https://www.reddit.com/r/ValueInvesting/.rss",
        "https://www.reddit.com/r/algotrading/.rss",
        "https://www.reddit.com/r/economy/.rss",
        "https://www.reddit.com/r/personalfinance/.rss",
        # ── Hacker News (finance / AI / startup news) ────────────────────────
        "https://news.ycombinator.com/rss",
        # ── Google News — macro / monetary policy (50-100 per query) ─────────
        "https://news.google.com/rss/search?q=federal+reserve+interest+rates&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=inflation+CPI+consumer+prices&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=GDP+economic+growth+recession&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=jobs+report+unemployment+payroll&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=central+bank+monetary+policy+rate+hike&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bond+yield+treasury+debt+deficit&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=dollar+index+forex+currency+exchange&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — equity markets ──────────────────────────────────────
        "https://news.google.com/rss/search?q=stock+market+earnings+S%26P500&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Nasdaq+Dow+Jones+index+rally&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=earnings+report+quarterly+results+EPS&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=IPO+SPAC+merger+acquisition+deal&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=short+selling+hedge+fund+activist+investor&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=dividend+buyback+stock+split+shareholder&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — banking & credit ────────────────────────────────────
        "https://news.google.com/rss/search?q=banking+financial+crisis+risk&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bank+failure+credit+risk+default&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=JPMorgan+Goldman+Sachs+Morgan+Stanley+bank&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=fintech+payments+digital+banking+neobank&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=credit+card+debt+consumer+lending+loan&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=SEC+FDIC+OCC+banking+regulation+enforcement&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — big tech ────────────────────────────────────────────
        "https://news.google.com/rss/search?q=Apple+AAPL+iPhone+earnings+revenue&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Microsoft+MSFT+Azure+cloud+AI+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Nvidia+NVDA+GPU+AI+chip+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Google+Alphabet+GOOGL+search+AI+revenue&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Amazon+AMZN+AWS+ecommerce+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Meta+Facebook+Instagram+advertising+revenue&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Tesla+TSLA+EV+electric+vehicle+Musk&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=AI+artificial+intelligence+regulation+model&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=semiconductor+chip+TSMC+Intel+foundry&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=antitrust+big+tech+regulation+monopoly&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — energy & commodities ───────────────────────────────
        "https://news.google.com/rss/search?q=oil+energy+commodities+OPEC+crude&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=natural+gas+LNG+energy+crisis+pipeline&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=gold+silver+copper+precious+metals+mining&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=renewable+energy+solar+wind+climate+ESG&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — crypto ──────────────────────────────────────────────
        "https://news.google.com/rss/search?q=bitcoin+BTC+price+ETF+crypto&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=ethereum+DeFi+stablecoin+blockchain+Web3&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=crypto+regulation+SEC+CFTC+exchange&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — geopolitics & trade ────────────────────────────────
        "https://news.google.com/rss/search?q=geopolitical+sanctions+trade+war+tariff&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=China+economy+trade+tariff+Xi+Jinping&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Europe+eurozone+ECB+recession+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Russia+Ukraine+war+sanctions+commodity&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=supply+chain+semiconductor+shortage+reshoring&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Middle+East+conflict+oil+supply+geopolitical&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=India+economy+rupee+growth+Modi&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Japan+BOJ+yen+yield+curve+economy&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — real estate & housing ───────────────────────────────
        "https://news.google.com/rss/search?q=real+estate+mortgage+housing+market+prices&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=commercial+real+estate+office+REIT+crisis&hl=en-US&gl=US&ceid=US:en",
        # ── Google News — emerging markets ────────────────────────────────────
        "https://news.google.com/rss/search?q=emerging+markets+currency+forex+devaluation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=private+equity+venture+capital+startup+funding&hl=en-US&gl=US&ceid=US:en",
    ])


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
# RSS feed scraper (no API key needed)
# ---------------------------------------------------------------------------

def scrape_rss(params: ScrapeParams) -> list[RawStory]:
    """
    Fetch headlines from public RSS feeds (Reuters, BBC, NYT, etc.).

    No API key required. No rate limits. Returns real live headlines.
    Feeds are fetched concurrently (ThreadPoolExecutor) so 90+ feeds
    complete in ~5-10 s instead of 90-180 s sequentially.
    """
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed. Run: pip install feedparser")
        return []

    from calendar import timegm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    since = datetime.now(timezone.utc) - timedelta(minutes=params.lookback_minutes)

    def _fetch_one(feed_url: str) -> list[RawStory]:
        feed_stories: list[RawStory] = []
        try:
            feed = feedparser.parse(feed_url)
            feed_name = (
                feed.feed.get("title", feed_url.split("/")[2])
                if feed.feed
                else feed_url.split("/")[2]
            )

            for entry in feed.entries[: params.max_per_source]:
                headline = (entry.get("title") or "").strip()
                body = (entry.get("summary") or entry.get("description") or "").strip()

                if not headline:
                    continue

                # Parse published timestamp
                published_at = time.time()
                raw_pp = getattr(entry, "published_parsed", None)
                if raw_pp:
                    try:
                        published_at = float(timegm(raw_pp))
                    except Exception:
                        pass

                # Filter by lookback window
                if datetime.fromtimestamp(published_at, tz=timezone.utc) < since:
                    continue

                link = entry.get("link", "")
                feed_stories.append(RawStory(
                    headline=headline,
                    body=body,
                    source=f"rss:{feed_name}",
                    url=link if isinstance(link, str) else "",
                    published_at=published_at,
                ))

            logger.debug("RSS [%s]: %d entries within window", feed_name, len(feed_stories))
        except Exception as exc:
            logger.error("RSS feed failed [%s]: %s", feed_url, exc)
        return feed_stories

    stories: list[RawStory] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_fetch_one, url): url for url in params.rss_feeds}
        for future in as_completed(futures):
            stories.extend(future.result())

    logger.info("RSS total: %d stories from %d feeds", len(stories), len(params.rss_feeds))
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
    if "rss" in params.sources:
        raw.extend(scrape_rss(params))

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

