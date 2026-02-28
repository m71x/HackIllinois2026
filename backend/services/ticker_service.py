"""
Ticker Service
==============
Resolves stock ticker symbols to rich company descriptions, then embeds
those descriptions so they can be semantically compared against the narrative
direction vectors already in ChromaDB.

Why rich text instead of raw ticker symbols
-------------------------------------------
Raw symbols ("JPM", "XOM") have no semantic content for an embedding model.
By building a text that includes company name, sector, industry, and business
description, the resulting vector lands near the narrative directions that
actually affect that company.

  JPM embedding  →  near "Regional banking stress", "Fed monetary tightening"
  XOM embedding  →  near "Energy supply shock", "OPEC production policy shift"
  NVDA embedding →  near "AI semiconductor demand surge", "China tech restrictions"

The embed text is constructed as:

    {name} ({symbol}) — {sector}, {industry}
    {business_description}

Caching
-------
yfinance calls are network-bound (~200ms each). Company descriptions are stable
over weeks, so we cache each lookup in-process with a 24-hour TTL.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

_CACHE_TTL_SECONDS = 86_400   # 24 hours


@dataclass
class TickerInfo:
    symbol: str
    name: str
    sector: str
    industry: str
    description: str       # longBusinessSummary from yfinance, truncated to 600 chars
    market_cap: Optional[float]
    embed_text: str        # pre-built string ready for embed_text()
    fetched_at: float = field(default_factory=time.time)


@dataclass
class _CacheEntry:
    info: TickerInfo
    fetched_at: float


_cache: dict[str, _CacheEntry] = {}
_cache_lock = threading.Lock()


def _is_stale(entry: _CacheEntry) -> bool:
    return (time.time() - entry.fetched_at) > _CACHE_TTL_SECONDS


def build_embed_text(symbol: str, name: str, sector: str, industry: str, description: str) -> str:
    """
    Construct a rich text string for a company that embeds semantically near
    relevant narrative directions.

    Template:
        {name} ({symbol}) — {sector}, {industry}
        {description}
    """
    header = f"{name} ({symbol}) — {sector}, {industry}"
    body = description.strip()
    return f"{header}\n{body}"


def get_ticker_info(symbol: str) -> TickerInfo:
    """
    Fetch company metadata for a ticker symbol via yfinance.
    Results are cached in-process for 24 hours.

    Raises
    ------
    ValueError
        If the symbol is not found or returns no usable data.
    """
    symbol = symbol.upper().strip()

    with _cache_lock:
        entry = _cache.get(symbol)
        if entry and not _is_stale(entry):
            return entry.info

    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    ticker = yf.Ticker(symbol)
    info = ticker.info

    if not info or not info.get("longName") and not info.get("shortName"):
        raise ValueError(f"No data found for ticker '{symbol}'. Check the symbol is valid.")

    name = info.get("longName") or info.get("shortName") or symbol
    sector = info.get("sector") or "Unknown Sector"
    industry = info.get("industry") or "Unknown Industry"
    raw_desc = info.get("longBusinessSummary") or ""

    # Truncate description to keep embed text focused and within model token limits
    description = raw_desc[:600]

    embed_text = build_embed_text(symbol, name, sector, industry, description)

    ticker_info = TickerInfo(
        symbol=symbol,
        name=name,
        sector=sector,
        industry=industry,
        description=description,
        market_cap=info.get("marketCap"),
        embed_text=embed_text,
        fetched_at=time.time(),
    )

    with _cache_lock:
        _cache[symbol] = _CacheEntry(info=ticker_info, fetched_at=ticker_info.fetched_at)

    return ticker_info


def get_ticker_info_batch(symbols: list[str]) -> dict[str, TickerInfo | Exception]:
    """
    Fetch multiple tickers. Returns a dict mapping symbol → TickerInfo or Exception.
    Errors per-ticker do not abort the whole batch.
    """
    results: dict[str, TickerInfo | Exception] = {}
    for symbol in symbols:
        try:
            results[symbol.upper()] = get_ticker_info(symbol)
        except Exception as e:
            results[symbol.upper()] = e
    return results


def invalidate_cache(symbol: str = None):
    """
    Evict a specific symbol from cache (or all entries if symbol is None).
    Useful for testing or after a corporate action.
    """
    with _cache_lock:
        if symbol:
            _cache.pop(symbol.upper(), None)
        else:
            _cache.clear()
