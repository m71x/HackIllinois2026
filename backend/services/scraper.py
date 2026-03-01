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

    Sources
    -------
    sources : list[str]
        Which sources to pull from. Any subset of ["newsapi", "rss"].

    Query overrides
    ---------------
    news_query : str
        Override the default NewsAPI keyword query.

    Other
    -----
    dry_run : bool
        If True, fetch stories but do NOT ingest them. Returns the list for inspection.
    """
    lookback_minutes: int = 60
    max_per_source: int = 50
    sources: list[str] = field(default_factory=lambda: ["newsapi", "rss"])

    news_query: str = (
        "economy OR inflation OR recession OR \"federal reserve\" OR \"interest rates\" "
        "OR \"stock market\" OR GDP OR \"trade war\" OR sanctions OR geopolitical "
        "OR \"supply chain\" OR \"central bank\" OR \"banking crisis\" OR \"credit risk\""
    )

    dry_run: bool = False

    rss_feeds: list[str] = field(default_factory=lambda: [
        # ── CNBC ─────────────────────────────────────────────────────────────
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.cnbc.com/id/15839135/device/rss/rss.html",
        "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "https://www.cnbc.com/id/10000115/device/rss/rss.html",
        "https://www.cnbc.com/id/15839069/device/rss/rss.html",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/10000108/device/rss/rss.html",
        # ── BBC ──────────────────────────────────────────────────────────────
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://finance.yahoo.com/news/rssindex",
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
        # ── Reddit (?limit=100 overrides the default 25-item cap) ───────────────
        "https://www.reddit.com/r/investing/.rss?limit=100",
        "https://www.reddit.com/r/stocks/.rss?limit=100",
        "https://www.reddit.com/r/Economics/.rss?limit=100",
        "https://www.reddit.com/r/wallstreetbets/.rss?limit=100",
        "https://www.reddit.com/r/finance/.rss?limit=100",
        "https://www.reddit.com/r/StockMarket/.rss?limit=100",
        "https://www.reddit.com/r/options/.rss?limit=100",
        "https://www.reddit.com/r/SecurityAnalysis/.rss?limit=100",
        "https://www.reddit.com/r/MacroEconomics/.rss?limit=100",
        "https://www.reddit.com/r/CryptoCurrency/.rss?limit=100",
        "https://www.reddit.com/r/financialindependence/.rss?limit=100",
        "https://www.reddit.com/r/ValueInvesting/.rss?limit=100",
        "https://www.reddit.com/r/algotrading/.rss?limit=100",
        "https://www.reddit.com/r/economy/.rss?limit=100",
        "https://www.reddit.com/r/personalfinance/.rss?limit=100",
        "https://news.ycombinator.com/rss",
        # ── Google News ───────────────────────────────────────────────────────
        "https://news.google.com/rss/search?q=federal+reserve+interest+rates&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=inflation+CPI+consumer+prices&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=GDP+economic+growth+recession&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=jobs+report+unemployment+payroll&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=central+bank+monetary+policy+rate+hike&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bond+yield+treasury+debt+deficit&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=dollar+index+forex+currency+exchange&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=stock+market+earnings+S%26P500&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Nasdaq+Dow+Jones+index+rally&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=earnings+report+quarterly+results+EPS&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=IPO+SPAC+merger+acquisition+deal&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=short+selling+hedge+fund+activist+investor&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=dividend+buyback+stock+split+shareholder&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=banking+financial+crisis+risk&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bank+failure+credit+risk+default&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=JPMorgan+Goldman+Sachs+Morgan+Stanley+bank&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=fintech+payments+digital+banking+neobank&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=credit+card+debt+consumer+lending+loan&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=SEC+FDIC+OCC+banking+regulation+enforcement&hl=en-US&gl=US&ceid=US:en",
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
        "https://news.google.com/rss/search?q=oil+energy+commodities+OPEC+crude&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=natural+gas+LNG+energy+crisis+pipeline&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=gold+silver+copper+precious+metals+mining&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=renewable+energy+solar+wind+climate+ESG&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bitcoin+BTC+price+ETF+crypto&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=ethereum+DeFi+stablecoin+blockchain+Web3&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=crypto+regulation+SEC+CFTC+exchange&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=geopolitical+sanctions+trade+war+tariff&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=China+economy+trade+tariff+Xi+Jinping&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Europe+eurozone+ECB+recession+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Russia+Ukraine+war+sanctions+commodity&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=supply+chain+semiconductor+shortage+reshoring&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Middle+East+conflict+oil+supply+geopolitical&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=India+economy+rupee+growth+Modi&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Japan+BOJ+yen+yield+curve+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=real+estate+mortgage+housing+market+prices&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=commercial+real+estate+office+REIT+crisis&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=emerging+markets+currency+forex+devaluation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=private+equity+venture+capital+startup+funding&hl=en-US&gl=US&ceid=US:en",

        # ── Reuters ───────────────────────────────────────────────────────────
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/Reuters/marketsNews",
        "https://feeds.reuters.com/reuters/technologyNews",
        "https://feeds.reuters.com/Reuters/worldNews",
        "https://feeds.reuters.com/reuters/companyNews",
        "https://feeds.reuters.com/reuters/environmentNews",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
        "https://feeds.npr.org/1006/rss.xml",
        "https://feeds.npr.org/1017/rss.xml",
        "https://abcnews.go.com/abcnews/moneyheadlines",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
        "https://www.reddit.com/r/dividends/.rss?limit=100",
        "https://www.reddit.com/r/ETFs/.rss?limit=100",
        "https://www.reddit.com/r/Bogleheads/.rss?limit=100",
        "https://www.reddit.com/r/realestateinvesting/.rss?limit=100",
        "https://www.reddit.com/r/smallbusiness/.rss?limit=100",
        "https://www.reddit.com/r/Entrepreneur/.rss?limit=100",
        "https://www.reddit.com/r/Gold/.rss?limit=100",
        "https://www.reddit.com/r/Bitcoin/.rss?limit=100",
        "https://www.reddit.com/r/ethereum/.rss?limit=100",
        "https://www.reddit.com/r/energy/.rss?limit=100",
        "https://www.reddit.com/r/RealEstate/.rss?limit=100",
        "https://www.reddit.com/r/fatFIRE/.rss?limit=100",
        "https://www.reddit.com/r/bonds/.rss?limit=100",
        "https://www.reddit.com/r/Commodities/.rss?limit=100",
        "https://news.google.com/rss/search?q=XLF+financial+sector+ETF+bank+interest+rates&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLK+technology+ETF+software+semiconductor&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLE+energy+ETF+oil+gas+renewable+sector&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLV+healthcare+ETF+pharma+biotech+sector&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLI+industrial+ETF+manufacturing+PMI+sector&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLY+consumer+discretionary+retail+spending+ETF&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLP+consumer+staples+food+beverage+grocery+ETF&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLU+utilities+power+grid+electricity+ETF&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLRE+REIT+real+estate+investment+trust+property&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLB+materials+mining+metals+copper+ETF&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XLC+communication+services+media+telecom+ETF&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=IWM+Russell+2000+small+cap+growth+ETF&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Berkshire+Hathaway+Warren+Buffett+BRK+investment&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=JPMorgan+Jamie+Dimon+bank+earnings+forecast&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Bank+of+America+Wells+Fargo+Citigroup+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Johnson+Pfizer+Merck+Abbott+pharma+drug+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=UnitedHealth+CVS+Humana+healthcare+insurance+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Exxon+Chevron+ConocoPhillips+XOM+CVX+oil+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Boeing+Lockheed+Raytheon+defense+aerospace+contract&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Ford+GM+Stellantis+automotive+EV+production+sales&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Walmart+Target+Home+Depot+retail+sales+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Disney+Warner+Netflix+streaming+media+entertainment&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Visa+Mastercard+PayPal+fintech+payment+transaction&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Salesforce+Oracle+SAP+enterprise+software+cloud&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=AT%26T+Verizon+T-Mobile+telecom+5G+wireless&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=AstraZeneca+Eli+Lilly+AbbVie+drug+biotech+clinical&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Caterpillar+Deere+industrials+machinery+equipment&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Costco+Nike+Starbucks+consumer+brand+retail+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Goldman+Sachs+Morgan+Stanley+investment+bank+deal&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=UPS+FedEx+logistics+shipping+supply+chain+delivery&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=AMD+Intel+Qualcomm+chip+processor+semiconductor+AI&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Coinbase+Binance+crypto+exchange+trading+volume&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Shopify+Square+Block+fintech+ecommerce+payment&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Airbnb+Uber+Lyft+DoorDash+gig+economy+platform&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=IMF+World+Bank+global+economy+debt+crisis+outlook&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=G7+G20+global+summit+trade+policy+agreement&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=tariff+trade+policy+White+House+executive+order&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=budget+deficit+spending+government+debt+ceiling&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=housing+mortgage+rates+home+prices+foreclosure&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=student+loan+debt+education+college+financing&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=insurance+catastrophe+climate+risk+reinsurance&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=pension+fund+sovereign+wealth+retirement+assets&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=commodity+agriculture+wheat+corn+food+prices+USDA&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=shipping+freight+container+port+supply+chain+cost&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=retail+sales+holiday+consumer+confidence+spending&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=manufacturing+orders+factory+production+industrial&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=labor+market+wages+hiring+layoffs+unemployment&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=healthcare+drug+pricing+reform+insurance+coverage&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=cybersecurity+ransomware+breach+hack+corporate+data&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=ESG+climate+carbon+credit+emission+sustainability&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=infrastructure+construction+bridge+road+spending&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=DOJ+FTC+antitrust+lawsuit+regulation+merger+review&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=mergers+acquisitions+M%26A+deal+buyout+takeover&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=hedge+fund+Bridgewater+Citadel+investor+portfolio&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=UK+economy+pound+sterling+Bank+of+England+rate&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Canada+economy+Bank+of+Canada+housing+CAD&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Australia+RBA+commodity+mining+AUD+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=China+yuan+property+crisis+Evergrande+debt&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=India+Nifty+BSE+Sensex+economy+RBI+growth&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Brazil+South+America+economy+BRL+inflation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Korea+semiconductor+Samsung+Hyundai+economy+KRW&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Germany+DAX+economy+manufacturing+recession+ECB&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Switzerland+SNB+franc+UBS+Credit+Suisse+banking&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Singapore+Asia+financial+hub+ASEAN+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Saudi+Arabia+OPEC+oil+sovereign+wealth+Vision2030&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Africa+emerging+market+investment+currency+growth&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Mexico+peso+USMCA+trade+economy+Banxico&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Turkey+lira+inflation+economy+central+bank+rate&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=France+CAC+ECB+bond+yield+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=OpenAI+ChatGPT+LLM+AI+funding+valuation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=cloud+computing+AWS+Azure+GCP+data+center+capex&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=SpaceX+space+satellite+Starlink+launch+NASA&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=autonomous+vehicle+robotics+automation+AI+factory&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=CBDC+digital+dollar+central+bank+crypto+currency&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=NFT+metaverse+gaming+Web3+blockchain+token&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Solana+Cardano+altcoin+DeFi+crypto+token+price&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=quantum+computing+biotech+nanotech+deep+tech&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=5G+network+infrastructure+telecom+wireless+spectrum&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=FDA+drug+approval+biotech+pharma+clinical+trial&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=cancer+treatment+oncology+immunotherapy+drug&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=GLP-1+Ozempic+Wegovy+obesity+weight+loss+drug&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=genomics+CRISPR+gene+therapy+biotech+pipeline&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=solar+wind+battery+storage+clean+energy+IRA&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=nuclear+energy+reactor+SMR+uranium+power+plant&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=EV+charging+infrastructure+lithium+battery+grid&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=hydrogen+fuel+cell+green+energy+decarbonization&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=carbon+capture+CCS+emissions+climate+net+zero&hl=en-US&gl=US&ceid=US:en",

        # ── Additional sources ────────────────────────────────────────────────
        "https://feeds.feedburner.com/zerohedge/feed",
        "https://wolfstreet.com/feed/",
        "https://feeds.feedburner.com/calculatedrisk",
        "https://www.bls.gov/feed/news.rss",
        "https://www.imf.org/en/News/rss",
        "https://www.bankofengland.co.uk/rss/news",
        "https://www.rba.gov.au/rss/rss-cb-speeches.xml",
        "https://economictimes.indiatimes.com/markets/rss.cms",
        "https://www.fxstreet.com/rss/news",
        "https://www.valuewalk.com/feed/",
        "https://www.nasdaq.com/feed/rssoutbound?category=Earnings",
        "https://www.nasdaq.com/feed/rssoutbound?category=Dividends",
        "https://www.nasdaq.com/feed/rssoutbound?category=IPOs",
        "https://www.nasdaq.com/feed/rssoutbound?category=Economic+Events",
        "https://www.bis.org/rss.htm",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",

        "https://news.google.com/rss/search?q=PCE+core+inflation+deflator+personal+spending&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=ISM+PMI+manufacturing+services+business+activity&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=producer+price+index+PPI+wholesale+inflation+cost&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=VIX+volatility+options+fear+index+put+call+ratio&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=credit+spread+high+yield+junk+bond+investment+grade&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=data+center+hyperscaler+GPU+power+demand+AI+capex&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=SaaS+cloud+ARR+revenue+growth+software+valuation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=airline+aviation+travel+Delta+United+American+capacity&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=luxury+goods+LVMH+Hermes+Gucci+earnings+consumer&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=copper+aluminum+zinc+nickel+base+metals+commodity&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=wheat+corn+soybeans+agriculture+food+commodity+USDA&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Taiwan+Strait+TSMC+semiconductor+chip+geopolitical&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Southeast+Asia+Vietnam+Indonesia+ASEAN+investment+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Argentina+Brazil+Latin+America+economy+currency+inflation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=EU+AI+Act+regulation+artificial+intelligence+policy+governance&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=AI+agent+agentic+robotics+automation+model+inference&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=biotech+pharma+M%26A+acquisition+drug+pipeline+deal&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bank+M%26A+acquisition+consolidation+regional+community+bank&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=SOFR+repo+money+market+funding+liquidity+commercial+paper&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=gold+mining+Barrick+Newmont+precious+metal+reserves&hl=en-US&gl=US&ceid=US:en",

        # ── Yahoo Finance per-ticker RSS ──────────────────────────────────────
        "https://finance.yahoo.com/rss/headline?s=AAPL",
        "https://finance.yahoo.com/rss/headline?s=MSFT",
        "https://finance.yahoo.com/rss/headline?s=NVDA",
        "https://finance.yahoo.com/rss/headline?s=AMZN",
        "https://finance.yahoo.com/rss/headline?s=GOOGL",
        "https://finance.yahoo.com/rss/headline?s=GOOG",
        "https://finance.yahoo.com/rss/headline?s=META",
        "https://finance.yahoo.com/rss/headline?s=TSLA",
        "https://finance.yahoo.com/rss/headline?s=AVGO",
        "https://finance.yahoo.com/rss/headline?s=ORCL",

        "https://finance.yahoo.com/rss/headline?s=BRK-B",
        "https://finance.yahoo.com/rss/headline?s=JPM",
        "https://finance.yahoo.com/rss/headline?s=V",
        "https://finance.yahoo.com/rss/headline?s=MA",
        "https://finance.yahoo.com/rss/headline?s=BAC",
        "https://finance.yahoo.com/rss/headline?s=GS",
        "https://finance.yahoo.com/rss/headline?s=MS",
        "https://finance.yahoo.com/rss/headline?s=WFC",
        "https://finance.yahoo.com/rss/headline?s=SPGI",
        "https://finance.yahoo.com/rss/headline?s=AXP",
        "https://finance.yahoo.com/rss/headline?s=BLK",
        "https://finance.yahoo.com/rss/headline?s=C",
        "https://finance.yahoo.com/rss/headline?s=CB",
        "https://finance.yahoo.com/rss/headline?s=CME",
        "https://finance.yahoo.com/rss/headline?s=MCO",
        "https://finance.yahoo.com/rss/headline?s=USB",
        "https://finance.yahoo.com/rss/headline?s=PNC",
        "https://finance.yahoo.com/rss/headline?s=TRV",
        "https://finance.yahoo.com/rss/headline?s=AON",
        "https://finance.yahoo.com/rss/headline?s=PYPL",
        "https://finance.yahoo.com/rss/headline?s=LLY",
        "https://finance.yahoo.com/rss/headline?s=UNH",
        "https://finance.yahoo.com/rss/headline?s=JNJ",
        "https://finance.yahoo.com/rss/headline?s=ABBV",
        "https://finance.yahoo.com/rss/headline?s=MRK",
        "https://finance.yahoo.com/rss/headline?s=TMO",
        "https://finance.yahoo.com/rss/headline?s=ABT",
        "https://finance.yahoo.com/rss/headline?s=PFE",
        "https://finance.yahoo.com/rss/headline?s=AMGN",
        "https://finance.yahoo.com/rss/headline?s=ISRG",
        "https://finance.yahoo.com/rss/headline?s=SYK",
        "https://finance.yahoo.com/rss/headline?s=VRTX",
        "https://finance.yahoo.com/rss/headline?s=GILD",
        "https://finance.yahoo.com/rss/headline?s=REGN",
        "https://finance.yahoo.com/rss/headline?s=BMY",
        "https://finance.yahoo.com/rss/headline?s=BSX",
        "https://finance.yahoo.com/rss/headline?s=ZTS",
        "https://finance.yahoo.com/rss/headline?s=CVS",
        "https://finance.yahoo.com/rss/headline?s=CI",
        "https://finance.yahoo.com/rss/headline?s=ELV",
        "https://finance.yahoo.com/rss/headline?s=HUM",
        "https://finance.yahoo.com/rss/headline?s=XOM",
        "https://finance.yahoo.com/rss/headline?s=CVX",
        "https://finance.yahoo.com/rss/headline?s=COP",
        "https://finance.yahoo.com/rss/headline?s=SLB",
        "https://finance.yahoo.com/rss/headline?s=EOG",
        "https://finance.yahoo.com/rss/headline?s=MPC",
        "https://finance.yahoo.com/rss/headline?s=PSX",
        "https://finance.yahoo.com/rss/headline?s=VLO",
        "https://finance.yahoo.com/rss/headline?s=PXD",
        "https://finance.yahoo.com/rss/headline?s=OXY",
        "https://finance.yahoo.com/rss/headline?s=KMI",
        "https://finance.yahoo.com/rss/headline?s=WMB",
        "https://finance.yahoo.com/rss/headline?s=COST",
        "https://finance.yahoo.com/rss/headline?s=WMT",
        "https://finance.yahoo.com/rss/headline?s=HD",
        "https://finance.yahoo.com/rss/headline?s=MCD",
        "https://finance.yahoo.com/rss/headline?s=NKE",
        "https://finance.yahoo.com/rss/headline?s=SBUX",
        "https://finance.yahoo.com/rss/headline?s=TGT",
        "https://finance.yahoo.com/rss/headline?s=LOW",
        "https://finance.yahoo.com/rss/headline?s=TJX",
        "https://finance.yahoo.com/rss/headline?s=BKNG",
        "https://finance.yahoo.com/rss/headline?s=MAR",
        "https://finance.yahoo.com/rss/headline?s=HLT",
        "https://finance.yahoo.com/rss/headline?s=ABNB",
        "https://finance.yahoo.com/rss/headline?s=GM",
        "https://finance.yahoo.com/rss/headline?s=F",
        "https://finance.yahoo.com/rss/headline?s=GE",
        "https://finance.yahoo.com/rss/headline?s=CAT",
        "https://finance.yahoo.com/rss/headline?s=RTX",
        "https://finance.yahoo.com/rss/headline?s=HON",
        "https://finance.yahoo.com/rss/headline?s=UNP",
        "https://finance.yahoo.com/rss/headline?s=BA",
        "https://finance.yahoo.com/rss/headline?s=LMT",
        "https://finance.yahoo.com/rss/headline?s=DE",
        "https://finance.yahoo.com/rss/headline?s=MMM",
        "https://finance.yahoo.com/rss/headline?s=ETN",
        "https://finance.yahoo.com/rss/headline?s=EMR",
        "https://finance.yahoo.com/rss/headline?s=ITW",
        "https://finance.yahoo.com/rss/headline?s=NOC",
        "https://finance.yahoo.com/rss/headline?s=GD",
        "https://finance.yahoo.com/rss/headline?s=UPS",
        "https://finance.yahoo.com/rss/headline?s=FDX",
        "https://finance.yahoo.com/rss/headline?s=AMD",
        "https://finance.yahoo.com/rss/headline?s=INTC",
        "https://finance.yahoo.com/rss/headline?s=QCOM",
        "https://finance.yahoo.com/rss/headline?s=TXN",
        "https://finance.yahoo.com/rss/headline?s=AMAT",
        "https://finance.yahoo.com/rss/headline?s=LRCX",
        "https://finance.yahoo.com/rss/headline?s=KLAC",
        "https://finance.yahoo.com/rss/headline?s=MU",
        "https://finance.yahoo.com/rss/headline?s=CSCO",
        "https://finance.yahoo.com/rss/headline?s=ADBE",
        "https://finance.yahoo.com/rss/headline?s=CRM",
        "https://finance.yahoo.com/rss/headline?s=NOW",
        "https://finance.yahoo.com/rss/headline?s=INTU",
        "https://finance.yahoo.com/rss/headline?s=PANW",
        "https://finance.yahoo.com/rss/headline?s=SNOW",
        "https://finance.yahoo.com/rss/headline?s=PLTR",
        "https://finance.yahoo.com/rss/headline?s=IBM",
        "https://finance.yahoo.com/rss/headline?s=ACN",
        "https://finance.yahoo.com/rss/headline?s=ADI",
        "https://finance.yahoo.com/rss/headline?s=APH",
        "https://finance.yahoo.com/rss/headline?s=VZ",
        "https://finance.yahoo.com/rss/headline?s=T",
        "https://finance.yahoo.com/rss/headline?s=TMUS",
        "https://finance.yahoo.com/rss/headline?s=NFLX",
        "https://finance.yahoo.com/rss/headline?s=DIS",
        "https://finance.yahoo.com/rss/headline?s=CMCSA",
        "https://finance.yahoo.com/rss/headline?s=WBD",
        "https://finance.yahoo.com/rss/headline?s=PARA",
        "https://finance.yahoo.com/rss/headline?s=LIN",
        "https://finance.yahoo.com/rss/headline?s=APD",
        "https://finance.yahoo.com/rss/headline?s=SHW",
        "https://finance.yahoo.com/rss/headline?s=ECL",
        "https://finance.yahoo.com/rss/headline?s=NEE",
        "https://finance.yahoo.com/rss/headline?s=SO",
        "https://finance.yahoo.com/rss/headline?s=DUK",
        "https://finance.yahoo.com/rss/headline?s=D",
        "https://finance.yahoo.com/rss/headline?s=AEP",
        "https://finance.yahoo.com/rss/headline?s=EXC",
        "https://finance.yahoo.com/rss/headline?s=PLD",
        "https://finance.yahoo.com/rss/headline?s=AMT",
        "https://finance.yahoo.com/rss/headline?s=EQIX",
        "https://finance.yahoo.com/rss/headline?s=CCI",
        "https://finance.yahoo.com/rss/headline?s=SPG",
        "https://finance.yahoo.com/rss/headline?s=O",
        "https://finance.yahoo.com/rss/headline?s=PG",
        "https://finance.yahoo.com/rss/headline?s=KO",
        "https://finance.yahoo.com/rss/headline?s=PEP",
        "https://finance.yahoo.com/rss/headline?s=PM",
        "https://finance.yahoo.com/rss/headline?s=MO",
        "https://finance.yahoo.com/rss/headline?s=MDLZ",
        "https://finance.yahoo.com/rss/headline?s=CL",
        "https://finance.yahoo.com/rss/headline?s=KHC",
        "https://finance.yahoo.com/rss/headline?s=GIS",
        "https://finance.yahoo.com/rss/headline?s=K",
        "https://finance.yahoo.com/rss/headline?s=SHOP",
        "https://finance.yahoo.com/rss/headline?s=COIN",
        "https://finance.yahoo.com/rss/headline?s=MSTR",
        "https://finance.yahoo.com/rss/headline?s=HOOD",
        "https://finance.yahoo.com/rss/headline?s=SOFI",
        "https://finance.yahoo.com/rss/headline?s=RIVN",
        "https://finance.yahoo.com/rss/headline?s=LCID",
        "https://finance.yahoo.com/rss/headline?s=NIO",
        "https://finance.yahoo.com/rss/headline?s=BIDU",
        "https://finance.yahoo.com/rss/headline?s=BABA",
        "https://finance.yahoo.com/rss/headline?s=JD",
        "https://finance.yahoo.com/rss/headline?s=PDD",
        "https://finance.yahoo.com/rss/headline?s=TSM",
        "https://finance.yahoo.com/rss/headline?s=ASML",
        "https://finance.yahoo.com/rss/headline?s=SAP",
        "https://finance.yahoo.com/rss/headline?s=NVO",
        "https://finance.yahoo.com/rss/headline?s=SHEL",
        "https://finance.yahoo.com/rss/headline?s=BP",
        "https://finance.yahoo.com/rss/headline?s=TTE",
        "https://finance.yahoo.com/rss/headline?s=RIO",
        "https://finance.yahoo.com/rss/headline?s=BHP",
        "https://finance.yahoo.com/rss/headline?s=MET",
        "https://finance.yahoo.com/rss/headline?s=PRU",
        "https://finance.yahoo.com/rss/headline?s=AIG",
        "https://finance.yahoo.com/rss/headline?s=ALL",
        "https://finance.yahoo.com/rss/headline?s=AFL",
        "https://finance.yahoo.com/rss/headline?s=STT",
        "https://finance.yahoo.com/rss/headline?s=BK",
        "https://finance.yahoo.com/rss/headline?s=SCHW",
        "https://finance.yahoo.com/rss/headline?s=TROW",
        "https://finance.yahoo.com/rss/headline?s=IVZ",
        "https://finance.yahoo.com/rss/headline?s=EBAY",
        "https://finance.yahoo.com/rss/headline?s=ETSY",
        "https://finance.yahoo.com/rss/headline?s=LULU",
        "https://finance.yahoo.com/rss/headline?s=ORLY",
        "https://finance.yahoo.com/rss/headline?s=AZO",
        "https://finance.yahoo.com/rss/headline?s=YUM",
        "https://finance.yahoo.com/rss/headline?s=DPZ",
        "https://finance.yahoo.com/rss/headline?s=EL",
        "https://finance.yahoo.com/rss/headline?s=ULTA",
        "https://finance.yahoo.com/rss/headline?s=RH",
        "https://finance.yahoo.com/rss/headline?s=EXPE",
        "https://finance.yahoo.com/rss/headline?s=DXCM",
        "https://finance.yahoo.com/rss/headline?s=IQV",
        "https://finance.yahoo.com/rss/headline?s=MCK",
        "https://finance.yahoo.com/rss/headline?s=CAH",
        "https://finance.yahoo.com/rss/headline?s=CNC",
        "https://finance.yahoo.com/rss/headline?s=HCA",
        "https://finance.yahoo.com/rss/headline?s=MRNA",
        "https://finance.yahoo.com/rss/headline?s=BIIB",
        "https://finance.yahoo.com/rss/headline?s=ALNY",
        "https://finance.yahoo.com/rss/headline?s=SMCI",
        # Energy (more)
        "https://finance.yahoo.com/rss/headline?s=DVN",
        "https://finance.yahoo.com/rss/headline?s=HAL",
        "https://finance.yahoo.com/rss/headline?s=BKR",
        "https://finance.yahoo.com/rss/headline?s=MRO",
        "https://finance.yahoo.com/rss/headline?s=LNG",
        "https://finance.yahoo.com/rss/headline?s=OKE",
        "https://finance.yahoo.com/rss/headline?s=FANG",
        "https://finance.yahoo.com/rss/headline?s=DAL",
        "https://finance.yahoo.com/rss/headline?s=UAL",
        "https://finance.yahoo.com/rss/headline?s=AAL",
        "https://finance.yahoo.com/rss/headline?s=LUV",
        "https://finance.yahoo.com/rss/headline?s=CCL",
        "https://finance.yahoo.com/rss/headline?s=RCL",
        "https://finance.yahoo.com/rss/headline?s=NCLH",
        # Tech — growth/SaaS/cybersecurity
        "https://finance.yahoo.com/rss/headline?s=UBER",
        "https://finance.yahoo.com/rss/headline?s=LYFT",
        "https://finance.yahoo.com/rss/headline?s=SNAP",
        "https://finance.yahoo.com/rss/headline?s=SPOT",
        "https://finance.yahoo.com/rss/headline?s=RBLX",
        "https://finance.yahoo.com/rss/headline?s=DDOG",
        "https://finance.yahoo.com/rss/headline?s=CRWD",
        "https://finance.yahoo.com/rss/headline?s=NET",
        "https://finance.yahoo.com/rss/headline?s=FTNT",
        "https://finance.yahoo.com/rss/headline?s=ZM",
        "https://finance.yahoo.com/rss/headline?s=WDAY",
        "https://finance.yahoo.com/rss/headline?s=OKTA",
        "https://finance.yahoo.com/rss/headline?s=MDB",
        "https://finance.yahoo.com/rss/headline?s=ZS",
        "https://finance.yahoo.com/rss/headline?s=GTLB",
        "https://finance.yahoo.com/rss/headline?s=HUBS",
        "https://finance.yahoo.com/rss/headline?s=TWLO",
        "https://finance.yahoo.com/rss/headline?s=FITB",
        "https://finance.yahoo.com/rss/headline?s=KEY",
        "https://finance.yahoo.com/rss/headline?s=HBAN",
        "https://finance.yahoo.com/rss/headline?s=RF",
        "https://finance.yahoo.com/rss/headline?s=CFG",
        "https://finance.yahoo.com/rss/headline?s=MTB",
        "https://finance.yahoo.com/rss/headline?s=NTRS",
        "https://finance.yahoo.com/rss/headline?s=EFX",
        "https://finance.yahoo.com/rss/headline?s=FDS",
        "https://finance.yahoo.com/rss/headline?s=ICE",
        "https://finance.yahoo.com/rss/headline?s=NDAQ",
        # Industrials / Transport (more)
        "https://finance.yahoo.com/rss/headline?s=CSX",
        "https://finance.yahoo.com/rss/headline?s=NSC",
        "https://finance.yahoo.com/rss/headline?s=ODFL",
        "https://finance.yahoo.com/rss/headline?s=CHRW",
        "https://finance.yahoo.com/rss/headline?s=JBHT",
        "https://finance.yahoo.com/rss/headline?s=GWW",
        "https://finance.yahoo.com/rss/headline?s=ROK",
        "https://finance.yahoo.com/rss/headline?s=IR",
        "https://finance.yahoo.com/rss/headline?s=CARR",
        "https://finance.yahoo.com/rss/headline?s=OTIS",
        "https://finance.yahoo.com/rss/headline?s=WELL",
        "https://finance.yahoo.com/rss/headline?s=VTR",
        "https://finance.yahoo.com/rss/headline?s=EQR",
        "https://finance.yahoo.com/rss/headline?s=AVB",
        "https://finance.yahoo.com/rss/headline?s=PSA",
        "https://finance.yahoo.com/rss/headline?s=EXR",
        "https://finance.yahoo.com/rss/headline?s=HST",
        "https://finance.yahoo.com/rss/headline?s=PCG",
        "https://finance.yahoo.com/rss/headline?s=EIX",
        "https://finance.yahoo.com/rss/headline?s=WEC",
        "https://finance.yahoo.com/rss/headline?s=ETR",
        "https://finance.yahoo.com/rss/headline?s=FE",
        "https://finance.yahoo.com/rss/headline?s=PPL",
        "https://finance.yahoo.com/rss/headline?s=GOLD",
        "https://finance.yahoo.com/rss/headline?s=NEM",
        "https://finance.yahoo.com/rss/headline?s=FCX",
        "https://finance.yahoo.com/rss/headline?s=AA",
        "https://finance.yahoo.com/rss/headline?s=NUE",
        "https://finance.yahoo.com/rss/headline?s=CF",
        "https://finance.yahoo.com/rss/headline?s=MOS",
        "https://finance.yahoo.com/rss/headline?s=WPM",
        "https://finance.yahoo.com/rss/headline?s=VALE",
        # International ADRs
        "https://finance.yahoo.com/rss/headline?s=TM",
        "https://finance.yahoo.com/rss/headline?s=SONY",
        "https://finance.yahoo.com/rss/headline?s=HMC",
        "https://finance.yahoo.com/rss/headline?s=NVS",
        "https://finance.yahoo.com/rss/headline?s=DEO",
        "https://finance.yahoo.com/rss/headline?s=BTI",
        "https://finance.yahoo.com/rss/headline?s=UL",
        "https://finance.yahoo.com/rss/headline?s=NSRGY",
        "https://finance.yahoo.com/rss/headline?s=LVMUY",
        "https://finance.yahoo.com/rss/headline?s=SE",
        "https://finance.yahoo.com/rss/headline?s=GRAB",
        "https://finance.yahoo.com/rss/headline?s=MELI",
        "https://finance.yahoo.com/rss/headline?s=NU",
        "https://finance.yahoo.com/rss/headline?s=ITUB",
        "https://finance.yahoo.com/rss/headline?s=INFY",
        "https://finance.yahoo.com/rss/headline?s=WIT",
        "https://finance.yahoo.com/rss/headline?s=HDB",
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


_cache = DeduplicatingCache(maxsize=100_000)


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

        if not headline or headline == "[Removed]":
            continue

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

# ---------------------------------------------------------------------------
# RSS feed scraper (no API key needed)
# ---------------------------------------------------------------------------

def _fetch_feed(feed_url: str, params: ScrapeParams, since: datetime) -> list[RawStory]:
    """
    Fetch and parse a single RSS feed.  Returns raw (non-deduplicated) stories
    within the lookback window.  Module-level so it can be reused by both
    scrape_rss() and scrape_rss_streaming().
    """
    try:
        import feedparser
    except ImportError:
        return []

    from calendar import timegm

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

            published_at = time.time()
            raw_pp = getattr(entry, "published_parsed", None)
            if raw_pp:
                try:
                    published_at = float(timegm(raw_pp))
                except Exception:
                    pass

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


def scrape_rss(params: ScrapeParams) -> list[RawStory]:
    """
    Fetch headlines from all RSS feeds concurrently and return the full list.
    Used by the regular poll cycle.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    since = datetime.now(timezone.utc) - timedelta(minutes=params.lookback_minutes)

    stories: list[RawStory] = []
    with ThreadPoolExecutor(max_workers=256) as pool:
        futures = {pool.submit(_fetch_feed, url, params, since): url for url in params.rss_feeds}
        for future in as_completed(futures):
            stories.extend(future.result())

    logger.info("RSS total: %d stories from %d feeds", len(stories), len(params.rss_feeds))
    return stories


def scrape_rss_streaming(params: ScrapeParams):
    """
    Generator version of scrape_rss.

    Yields deduplicated story batches as each feed completes (as_completed),
    rather than waiting for all feeds to finish.  Callers can start processing
    the first batch while the remaining feeds are still in-flight.

    Used by bulk_ingest() to pipeline scraping → embedding → routing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    since = datetime.now(timezone.utc) - timedelta(minutes=params.lookback_minutes)

    with ThreadPoolExecutor(max_workers=256) as pool:
        futures = {pool.submit(_fetch_feed, url, params, since): url for url in params.rss_feeds}
        for future in as_completed(futures):
            raw = future.result()
            fresh = []
            for story in raw:
                if not _cache.is_seen(story.headline, story.body):
                    _cache.mark_seen(story.headline, story.body)
                    fresh.append(story)
            if fresh:
                yield fresh


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
    if "rss" in params.sources:
        raw.extend(scrape_rss(params))

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

