"""Tests for SEC EDGAR insider trades collector."""

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ck_trading.collectors.insider_trades import InsiderTradesCollector


@pytest.fixture
def collector():
    return InsiderTradesCollector(timeout=5.0)


@pytest.fixture
def sample_tickers_json():
    """SEC company_tickers.json format."""
    return {
        "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": "789019", "ticker": "MSFT", "title": "Microsoft Corp"},
    }


@pytest.fixture
def sample_company_filing():
    """SEC CIK submissions JSON with Form 4 filings."""
    return {
        "cik": "320193",
        "name": "Tim Cook",
        "filings": {
            "recent": {
                "form": ["4", "10-K", "4", "8-K", "4/A"],
                "filingDate": [
                    "2026-03-01",
                    "2026-02-15",
                    "2026-02-01",
                    "2026-01-15",
                    "2026-01-10",
                ],
                "primaryDocument": [
                    "doc1.xml",
                    "doc2.htm",
                    "doc3.xml",
                    "doc4.htm",
                    "doc5.xml",
                ],
            }
        },
    }


class TestInsiderTradesCollector:
    def test_collect_parses_form4(
        self, collector, sample_tickers_json, sample_company_filing
    ):
        """Mock SEC response and verify correct columns."""
        mock_tickers_resp = MagicMock()
        mock_tickers_resp.status_code = 200
        mock_tickers_resp.json.return_value = sample_tickers_json
        mock_tickers_resp.raise_for_status = MagicMock()

        mock_filing_resp = MagicMock()
        mock_filing_resp.status_code = 200
        mock_filing_resp.json.return_value = sample_company_filing
        mock_filing_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)

            def side_effect(url, **kwargs):
                if "company_tickers" in url:
                    return mock_tickers_resp
                return mock_filing_resp

            mock_client.get.side_effect = side_effect
            mock_client_cls.return_value = mock_client

            df = collector.collect(["AAPL"], days_back=365)

        assert isinstance(df, pl.DataFrame)
        expected_cols = {
            "ticker", "date", "insider_name", "title",
            "transaction_type", "shares", "price", "value",
        }
        assert set(df.columns) == expected_cols
        # Should find Form 4 and 4/A filings (3 total)
        assert len(df) == 3
        assert all(t == "AAPL" for t in df["ticker"].to_list())

    def test_collect_no_filings(self, collector, sample_tickers_json):
        """No Form 4 filings -> empty DataFrame."""
        empty_filing = {
            "cik": "320193",
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K"],
                    "filingDate": ["2024-03-01", "2024-02-01", "2024-01-01"],
                    "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                }
            },
        }

        mock_tickers_resp = MagicMock()
        mock_tickers_resp.status_code = 200
        mock_tickers_resp.json.return_value = sample_tickers_json
        mock_tickers_resp.raise_for_status = MagicMock()

        mock_filing_resp = MagicMock()
        mock_filing_resp.status_code = 200
        mock_filing_resp.json.return_value = empty_filing
        mock_filing_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)

            def side_effect(url, **kwargs):
                if "company_tickers" in url:
                    return mock_tickers_resp
                return mock_filing_resp

            mock_client.get.side_effect = side_effect
            mock_client_cls.return_value = mock_client

            df = collector.collect(["AAPL"], days_back=365)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_user_agent_header(self, collector):
        """Verify User-Agent is set in requests."""
        assert "User-Agent" in collector.HEADERS
        assert collector.HEADERS["User-Agent"] != ""
        assert "@" in collector.HEADERS["User-Agent"]  # Should contain email

    def test_collect_empty_tickers(self, collector):
        """Empty tickers list -> empty DataFrame without API call."""
        df = collector.collect([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_collect_unknown_ticker(self, collector, sample_tickers_json):
        """Unknown ticker (no CIK found) -> empty DataFrame."""
        mock_tickers_resp = MagicMock()
        mock_tickers_resp.status_code = 200
        mock_tickers_resp.json.return_value = sample_tickers_json
        mock_tickers_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_tickers_resp
            mock_client_cls.return_value = mock_client

            df = collector.collect(["ZZZZ"], days_back=365)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
