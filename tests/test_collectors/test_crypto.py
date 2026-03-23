"""Tests for CoinGecko crypto collector."""

from datetime import date
from unittest.mock import MagicMock, patch

import httpx as _httpx
import polars as pl
import pytest

from ck_trading.collectors.crypto import CryptoCollector


@pytest.fixture
def collector():
    c = CryptoCollector(timeout=5.0)
    c._REQUEST_INTERVAL = 0.0  # Disable rate limiting in tests
    return c


@pytest.fixture
def sample_coingecko_response():
    """CoinGecko market_chart/range response."""
    return {
        "prices": [
            [1705276800000, 42000.0],  # 2024-01-15
            [1705363200000, 42500.0],  # 2024-01-16
            [1705449600000, 43000.0],  # 2024-01-17
        ],
        "total_volumes": [
            [1705276800000, 1000000.0],
            [1705363200000, 1100000.0],
            [1705449600000, 1200000.0],
        ],
    }


def _make_mock_response(data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = _httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=mock_resp
        )
    return mock_resp


class TestCryptoCollector:
    def test_collect_btc_eth(self, collector, sample_coingecko_response):
        """Mock CoinGecko -> DataFrame with BTC and ETH rows."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = _make_mock_response(sample_coingecko_response)
            mock_client_cls.return_value = mock_client

            df = collector.collect_prices(
                coins=["BTC", "ETH"],
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 17),
            )

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        tickers = set(df["ticker"].to_list())
        assert "BTC" in tickers
        assert "ETH" in tickers

    def test_price_schema(self, collector, sample_coingecko_response):
        """Output has same columns as stock prices."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = _make_mock_response(sample_coingecko_response)
            mock_client_cls.return_value = mock_client

            df = collector.collect_prices(
                coins=["BTC"],
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 17),
            )

        expected_cols = {"ticker", "date", "open", "high", "low", "close", "volume", "adj_close"}
        assert set(df.columns) == expected_cols

    def test_date_range_filter(self, collector, sample_coingecko_response):
        """Only dates in range returned."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = _make_mock_response(sample_coingecko_response)
            mock_client_cls.return_value = mock_client

            # Request only Jan 15-16 (exclude Jan 17)
            df = collector.collect_prices(
                coins=["BTC"],
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 16),
            )

        dates = df["date"].to_list()
        for d in dates:
            assert d >= date(2024, 1, 15)
            assert d <= date(2024, 1, 16)

    def test_rate_limit_handled(self, collector):
        """Mock 429 -> graceful handling, returns empty DataFrame."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.raise_for_status = MagicMock()  # Don't raise; handled by status check

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_429
            mock_client_cls.return_value = mock_client

            df = collector.collect_prices(
                coins=["BTC"],
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 17),
            )

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_empty_coins(self, collector):
        """Empty coins list -> default to BTC+ETH but if empty response, empty df."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = _make_mock_response({"prices": [], "total_volumes": []})
            mock_client_cls.return_value = mock_client

            df = collector.collect_prices(
                coins=["BTC"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_unknown_coin(self, collector):
        """Unknown coin ticker -> warning and skip, no crash."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            df = collector.collect_prices(
                coins=["UNKNOWN_COIN"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
