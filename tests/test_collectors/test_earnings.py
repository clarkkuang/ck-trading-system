"""Tests for earnings data collector."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from ck_trading.collectors.earnings import EarningsCollector


@pytest.fixture
def collector_no_fmp():
    return EarningsCollector(fmp_api_key=None)


@pytest.fixture
def collector_with_fmp():
    return EarningsCollector(fmp_api_key="test_key_123", timeout=5.0)


@pytest.fixture
def sample_earnings_dates():
    """Mock yfinance earnings_dates DataFrame."""
    idx = pd.DatetimeIndex(
        ["2024-01-25", "2023-10-26", "2023-07-27"],
        name="Earnings Date",
    )
    return pd.DataFrame(
        {
            "EPS Estimate": [2.10, 1.95, 1.82],
            "Reported EPS": [2.18, 1.98, 1.85],
            "Surprise(%)": [3.81, 1.54, 1.65],
        },
        index=idx,
    )


@pytest.fixture
def sample_fmp_response():
    return [
        {
            "symbol": "AAPL",
            "date": "2024-09-30",
            "estimatedEpsAvg": 6.55,
            "estimatedRevenueAvg": 395000000000,
            "numberAnalystEstimatedEps": 30,
        },
        {
            "symbol": "AAPL",
            "date": "2024-06-30",
            "estimatedEpsAvg": 6.30,
            "estimatedRevenueAvg": 380000000000,
            "numberAnalystEstimatedEps": 28,
        },
    ]


class TestEarningsCollector:
    def test_earnings_history_from_yfinance(
        self, collector_no_fmp, sample_earnings_dates
    ):
        """Mock yf.Ticker -> correct columns."""
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.return_value = sample_earnings_dates

        with patch("yfinance.Ticker", return_value=mock_ticker):
            df = collector_no_fmp.collect_earnings_history(["AAPL"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        expected_cols = {
            "ticker", "date", "eps_estimate", "eps_actual",
            "surprise_pct", "revenue_estimate", "revenue_actual",
        }
        assert set(df.columns) == expected_cols
        assert all(t == "AAPL" for t in df["ticker"].to_list())
        # Verify actual values
        row0 = df.sort("date", descending=True).row(0, named=True)
        assert row0["eps_estimate"] == pytest.approx(2.10)
        assert row0["eps_actual"] == pytest.approx(2.18)

    def test_empty_earnings(self, collector_no_fmp):
        """Ticker with no earnings data -> empty DataFrame."""
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            df = collector_no_fmp.collect_earnings_history(["UNKNOWN"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "ticker" in df.columns

    def test_fmp_not_called_without_key(self, collector_no_fmp):
        """fmp_api_key=None -> no FMP requests made."""
        with patch("httpx.Client") as mock_client_cls:
            df = collector_no_fmp.collect_analyst_estimates(["AAPL"])
            # httpx.Client should not have been instantiated
            mock_client_cls.assert_not_called()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_fmp_analyst_estimates(self, collector_with_fmp, sample_fmp_response):
        """Mock FMP API -> correct columns."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_fmp_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            df = collector_with_fmp.collect_analyst_estimates(["AAPL"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        expected_cols = {
            "ticker", "date", "estimated_eps",
            "estimated_revenue", "number_of_analysts",
        }
        assert set(df.columns) == expected_cols
        assert all(t == "AAPL" for t in df["ticker"].to_list())

    def test_earnings_history_handles_nan(self, collector_no_fmp):
        """NaN values in yfinance data -> converted to None."""
        import numpy as np

        idx = pd.DatetimeIndex(["2024-01-25"], name="Earnings Date")
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [np.nan],
                "Reported EPS": [np.nan],
                "Surprise(%)": [np.nan],
            },
            index=idx,
        )

        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.return_value = earnings_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            df = collector_no_fmp.collect_earnings_history(["AAPL"])

        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["eps_estimate"] is None
        assert row["eps_actual"] is None
        assert row["surprise_pct"] is None

    def test_collect_empty_tickers(self, collector_no_fmp):
        """Empty tickers list -> empty DataFrame."""
        df = collector_no_fmp.collect_earnings_history([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_fmp_empty_tickers(self, collector_with_fmp):
        """Empty tickers list -> empty DataFrame without API call."""
        df = collector_with_fmp.collect_analyst_estimates([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
