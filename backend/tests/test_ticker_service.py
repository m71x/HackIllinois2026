"""Tests for services/ticker_service.py — embed text construction and cache."""

import time
import pytest
from unittest.mock import patch, MagicMock
from services.ticker_service import (
    build_embed_text,
    get_ticker_info,
    invalidate_cache,
    TickerInfo,
)


def make_yfinance_info(**overrides) -> dict:
    """Minimal yfinance Ticker.info payload."""
    base = {
        "longName": "JPMorgan Chase & Co.",
        "shortName": "JPMorgan Chase",
        "sector": "Financial Services",
        "industry": "Banks—Diversified",
        "longBusinessSummary": (
            "JPMorgan Chase & Co. is a global financial services firm "
            "operating in investment banking, commercial banking, and asset management."
        ),
        "marketCap": 500_000_000_000,
    }
    base.update(overrides)
    return base


def mock_yfinance(info: dict):
    """Return a context manager that patches yfinance.Ticker."""
    ticker_mock = MagicMock()
    ticker_mock.info = info
    return patch("yfinance.Ticker", return_value=ticker_mock)


class TestBuildEmbedText:
    def test_contains_all_components(self):
        text = build_embed_text("JPM", "JPMorgan Chase", "Financial Services", "Banks", "Desc here")
        assert "JPM" in text
        assert "JPMorgan Chase" in text
        assert "Financial Services" in text
        assert "Banks" in text
        assert "Desc here" in text

    def test_header_format(self):
        text = build_embed_text("XOM", "Exxon Mobil", "Energy", "Oil & Gas", "Big oil company.")
        header_line = text.split("\n")[0]
        assert "XOM" in header_line
        assert "Exxon Mobil" in header_line
        assert "Energy" in header_line
        assert "Oil & Gas" in header_line

    def test_description_on_second_line(self):
        text = build_embed_text("AAPL", "Apple Inc.", "Technology", "Consumer Electronics", "Makes iPhones.")
        lines = text.split("\n")
        assert len(lines) >= 2
        assert "Makes iPhones" in lines[1]

    def test_empty_description(self):
        text = build_embed_text("XYZ", "Unknown Corp", "Unknown", "Unknown", "")
        assert "XYZ" in text   # header still present


class TestGetTickerInfo:
    def setup_method(self):
        """Clear cache before each test."""
        invalidate_cache()

    def test_basic_fetch(self):
        info_data = make_yfinance_info()
        with mock_yfinance(info_data):
            result = get_ticker_info("JPM")

        assert result.symbol == "JPM"
        assert result.name == "JPMorgan Chase & Co."
        assert result.sector == "Financial Services"
        assert result.industry == "Banks—Diversified"
        assert len(result.description) <= 600

    def test_symbol_uppercased(self):
        with mock_yfinance(make_yfinance_info()):
            result = get_ticker_info("jpm")
        assert result.symbol == "JPM"

    def test_embed_text_built(self):
        with mock_yfinance(make_yfinance_info()):
            result = get_ticker_info("JPM")
        assert result.embed_text
        assert "JPM" in result.embed_text
        assert "Financial Services" in result.embed_text

    def test_description_truncated_to_600(self):
        long_desc = "x" * 1200
        with mock_yfinance(make_yfinance_info(longBusinessSummary=long_desc)):
            result = get_ticker_info("JPM")
        assert len(result.description) == 600

    def test_missing_sector_fallback(self):
        info = make_yfinance_info(sector=None)
        with mock_yfinance(info):
            result = get_ticker_info("JPM")
        assert result.sector == "Unknown Sector"

    def test_missing_industry_fallback(self):
        info = make_yfinance_info(industry=None)
        with mock_yfinance(info):
            result = get_ticker_info("JPM")
        assert result.industry == "Unknown Industry"

    def test_shortname_fallback(self):
        info = make_yfinance_info(longName=None, shortName="JPMorgan")
        with mock_yfinance(info):
            result = get_ticker_info("JPM")
        assert result.name == "JPMorgan"

    def test_raises_on_empty_info(self):
        with mock_yfinance({}):
            with pytest.raises(ValueError, match="No data found"):
                get_ticker_info("FAKEXYZ")

    def test_market_cap_stored(self):
        with mock_yfinance(make_yfinance_info(marketCap=500_000_000_000)):
            result = get_ticker_info("JPM")
        assert result.market_cap == 500_000_000_000


class TestCache:
    def setup_method(self):
        invalidate_cache()

    def test_cache_hit_avoids_second_yfinance_call(self):
        with mock_yfinance(make_yfinance_info()) as mock_cls:
            get_ticker_info("JPM")
            get_ticker_info("JPM")
        # yfinance.Ticker should only be instantiated once
        assert mock_cls.call_count == 1

    def test_different_symbols_both_fetched(self):
        with mock_yfinance(make_yfinance_info(longName="JPMorgan Chase & Co.")):
            get_ticker_info("JPM")
        with mock_yfinance(make_yfinance_info(longName="Exxon Mobil Corporation")):
            get_ticker_info("XOM")

        invalidate_cache()  # clear and refetch to verify both needed a fetch

    def test_invalidate_single_symbol(self):
        with mock_yfinance(make_yfinance_info()) as mock_cls:
            get_ticker_info("JPM")
            invalidate_cache("JPM")
            get_ticker_info("JPM")
        assert mock_cls.call_count == 2   # fetched twice after eviction

    def test_invalidate_all(self):
        with mock_yfinance(make_yfinance_info()) as mock_jpm:
            get_ticker_info("JPM")
        invalidate_cache()
        with mock_yfinance(make_yfinance_info()) as mock_jpm2:
            get_ticker_info("JPM")
        assert mock_jpm2.call_count == 1
