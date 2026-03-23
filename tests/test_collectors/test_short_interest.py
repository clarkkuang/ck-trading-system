"""Tests for FINRA short interest collector."""

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ck_trading.collectors.short_interest import ShortInterestCollector


@pytest.fixture
def collector():
    return ShortInterestCollector(timeout=5.0)


@pytest.fixture
def sample_reg_sho_response():
    """Sample Reg SHO daily response – multiple facilities for same ticker/date."""
    return [
        {
            "reportingFacilityCode": "NQTRF",
            "totalParQuantity": 10000,
            "shortParQuantity": 3000,
            "marketCode": "Q",
            "tradeReportDate": "2024-01-15",
            "securitiesInformationProcessorSymbolIdentifier": "AAPL",
            "shortExemptParQuantity": 50,
        },
        {
            "reportingFacilityCode": "NYTRF",
            "totalParQuantity": 5000,
            "shortParQuantity": 2000,
            "marketCode": "N",
            "tradeReportDate": "2024-01-15",
            "securitiesInformationProcessorSymbolIdentifier": "AAPL",
            "shortExemptParQuantity": 10,
        },
        {
            "reportingFacilityCode": "NQTRF",
            "totalParQuantity": 8000,
            "shortParQuantity": 4000,
            "marketCode": "Q",
            "tradeReportDate": "2024-01-15",
            "securitiesInformationProcessorSymbolIdentifier": "MSFT",
            "shortExemptParQuantity": 0,
        },
    ]


def _make_mock_client(responses_by_call):
    """Build a mock httpx.Client that returns different responses per .post() call."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = responses_by_call
    return mock_client


def _ok_response(data):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _empty_response():
    resp = MagicMock()
    resp.status_code = 204
    resp.raise_for_status = MagicMock()
    return resp


class TestShortInterestCollector:
    def test_collect_parses_and_aggregates(self, collector, sample_reg_sho_response):
        """Mock FINRA regShoDaily and verify aggregation across facilities."""
        # Collector calls _fetch_reg_sho once per ticker
        aapl_data = [r for r in sample_reg_sho_response if r["securitiesInformationProcessorSymbolIdentifier"] == "AAPL"]
        msft_data = [r for r in sample_reg_sho_response if r["securitiesInformationProcessorSymbolIdentifier"] == "MSFT"]

        with patch("httpx.Client") as mock_client_cls:
            mock_client = _make_mock_client([_ok_response(aapl_data), _ok_response(msft_data)])
            mock_client_cls.return_value = mock_client

            df = collector.collect(["AAPL", "MSFT"])

        assert isinstance(df, pl.DataFrame)
        # Two tickers, one date each -> 2 rows after aggregation
        assert len(df) == 2
        expected_cols = {"ticker", "date", "short_volume", "total_volume", "short_exempt_volume", "short_ratio"}
        assert set(df.columns) == expected_cols

        # AAPL should have aggregated: short=3000+2000=5000, total=10000+5000=15000
        aapl_row = df.filter(pl.col("ticker") == "AAPL")
        assert aapl_row["short_volume"].item() == 5000
        assert aapl_row["total_volume"].item() == 15000
        assert abs(aapl_row["short_ratio"].item() - 5000 / 15000) < 0.001

    def test_collect_empty_response(self, collector):
        """API returns 204 (no data) -> empty DataFrame."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = _make_mock_client([_empty_response()])
            mock_client_cls.return_value = mock_client

            df = collector.collect(["AAPL"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "ticker" in df.columns

    def test_collect_api_error(self, collector):
        """API 500 -> returns empty DataFrame gracefully."""
        import httpx as _httpx

        err_response = MagicMock()
        err_response.status_code = 500
        err_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=err_response
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_client = _make_mock_client([err_response])
            mock_client_cls.return_value = mock_client

            df = collector.collect(["AAPL"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_tickers_filtered(self, collector, sample_reg_sho_response):
        """Only requested tickers appear in output."""
        # Return all data but only request AAPL
        with patch("httpx.Client") as mock_client_cls:
            mock_client = _make_mock_client([_ok_response(sample_reg_sho_response)])
            mock_client_cls.return_value = mock_client

            df = collector.collect(["AAPL"])

        assert set(df["ticker"].to_list()) == {"AAPL"}

    def test_collect_empty_tickers(self, collector):
        """Empty tickers list -> empty DataFrame without API call."""
        df = collector.collect([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_collect_with_dates(self, collector):
        """Date filters are passed in the payload."""
        data = [
            {
                "reportingFacilityCode": "NQTRF",
                "totalParQuantity": 1000,
                "shortParQuantity": 500,
                "marketCode": "Q",
                "tradeReportDate": "2024-03-01",
                "securitiesInformationProcessorSymbolIdentifier": "TSLA",
                "shortExemptParQuantity": 0,
            }
        ]

        with patch("httpx.Client") as mock_client_cls:
            mock_client = _make_mock_client([_ok_response(data)])
            mock_client_cls.return_value = mock_client

            df = collector.collect(
                ["TSLA"],
                start_date=date(2024, 3, 1),
                end_date=date(2024, 3, 31),
            )

            # Verify the payload had dateRangeFilters
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "dateRangeFilters" in payload
            assert payload["dateRangeFilters"][0]["startDate"] == "2024-03-01"

        assert len(df) == 1
        assert df["ticker"].item() == "TSLA"
