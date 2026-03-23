"""Tests for Quality Factor strategy."""

from datetime import date

import polars as pl
import pytest

from ck_trading.strategies.quality_factor import QualityFactorStrategy


def _make_fundamentals(tickers_data: list[dict]) -> pl.DataFrame:
    """Build a multi-period fundamentals DataFrame.

    Each dict has:
        ticker: str
        periods: list of dicts with keys:
            period_end, roe, debt_to_equity, net_income, market_cap
    """
    rows = []
    for td in tickers_data:
        ticker = td["ticker"]
        for p in td["periods"]:
            rows.append({
                "ticker": ticker,
                "market": "US",
                "period_end": p["period_end"],
                "report_date": p["period_end"],
                "roe": p.get("roe"),
                "debt_to_equity": p.get("debt_to_equity"),
                "net_income": p.get("net_income"),
                "market_cap": p.get("market_cap", 1e10),
                "pe_ratio": p.get("pe_ratio", 15.0),
                "revenue": 1e9,
                "total_assets": 5e9,
                "total_equity": 2e9,
            })
    return pl.DataFrame(rows)


@pytest.fixture
def empty_prices() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )


class TestQualityFactorStrategy:
    def test_properties(self):
        s = QualityFactorStrategy()
        assert s.name == "Quality Factor"
        assert s.rebalance_frequency == "quarterly"
        assert s.description

    def test_high_quality_selected(self, empty_prices):
        """Ticker with high ROE, low D/E, stable earnings is selected."""
        data = [
            {
                "ticker": "QUALITY",
                "periods": [
                    {"period_end": date(2020, 3, 31), "roe": 0.25, "debt_to_equity": 0.5,
                     "net_income": 100e6, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "roe": 0.25, "debt_to_equity": 0.5,
                     "net_income": 110e6, "market_cap": 5e9},
                    {"period_end": date(2020, 9, 30), "roe": 0.25, "debt_to_equity": 0.5,
                     "net_income": 120e6, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = QualityFactorStrategy(min_roe=0.15, max_debt_to_equity=1.0, top_n=20)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert not result.is_empty()
        assert "QUALITY" in result["ticker"].to_list()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}
        assert (result["signal_type"] == "BUY").all()

    def test_low_roe_filtered(self, empty_prices):
        """ROE below threshold is filtered out."""
        data = [
            {
                "ticker": "LOWROE",
                "periods": [
                    {"period_end": date(2020, 3, 31), "roe": 0.05, "debt_to_equity": 0.3,
                     "net_income": 50e6, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "roe": 0.05, "debt_to_equity": 0.3,
                     "net_income": 55e6, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = QualityFactorStrategy(min_roe=0.15)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "LOWROE" not in result["ticker"].to_list()

    def test_high_debt_filtered(self, empty_prices):
        """High D/E is filtered out."""
        data = [
            {
                "ticker": "HIGHDEBT",
                "periods": [
                    {"period_end": date(2020, 3, 31), "roe": 0.25, "debt_to_equity": 3.0,
                     "net_income": 100e6, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "roe": 0.25, "debt_to_equity": 3.0,
                     "net_income": 110e6, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = QualityFactorStrategy(max_debt_to_equity=1.0)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "HIGHDEBT" not in result["ticker"].to_list()

    def test_earnings_stability_scoring(self, empty_prices):
        """Stable growth ticker scores higher than volatile one."""
        data = [
            {
                "ticker": "STABLE",
                "periods": [
                    {"period_end": date(2020, 3, 31), "roe": 0.20, "debt_to_equity": 0.5,
                     "net_income": 100e6, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "roe": 0.20, "debt_to_equity": 0.5,
                     "net_income": 110e6, "market_cap": 5e9},
                    {"period_end": date(2020, 9, 30), "roe": 0.20, "debt_to_equity": 0.5,
                     "net_income": 121e6, "market_cap": 5e9},
                ],
            },
            {
                "ticker": "VOLATILE",
                "periods": [
                    {"period_end": date(2020, 3, 31), "roe": 0.20, "debt_to_equity": 0.5,
                     "net_income": 100e6, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "roe": 0.20, "debt_to_equity": 0.5,
                     "net_income": 50e6, "market_cap": 5e9},
                    {"period_end": date(2020, 9, 30), "roe": 0.20, "debt_to_equity": 0.5,
                     "net_income": 200e6, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = QualityFactorStrategy(min_roe=0.15, max_debt_to_equity=1.0, top_n=20)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert not result.is_empty()
        tickers = result["ticker"].to_list()
        if "STABLE" in tickers and "VOLATILE" in tickers:
            stable_score = result.filter(pl.col("ticker") == "STABLE")["score"][0]
            volatile_score = result.filter(pl.col("ticker") == "VOLATILE")["score"][0]
            assert stable_score > volatile_score, (
                f"STABLE ({stable_score}) should score higher than VOLATILE ({volatile_score})"
            )

    def test_single_period_handled(self, empty_prices):
        """Only 1 period means can't compute stability; ticker is excluded."""
        data = [
            {
                "ticker": "ONLYONE",
                "periods": [
                    {"period_end": date(2020, 6, 30), "roe": 0.25, "debt_to_equity": 0.3,
                     "net_income": 100e6, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = QualityFactorStrategy(min_periods=2)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "ONLYONE" not in result["ticker"].to_list()

    def test_empty_fundamentals(self, empty_prices):
        """Returns empty result for empty input."""
        s = QualityFactorStrategy()
        result = s.screen(empty_prices, pl.DataFrame(), date(2020, 12, 31))
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_market_cap_filter(self, empty_prices):
        """Small cap below threshold is filtered out."""
        data = [
            {
                "ticker": "SMALLCAP",
                "periods": [
                    {"period_end": date(2020, 3, 31), "roe": 0.25, "debt_to_equity": 0.3,
                     "net_income": 10e6, "market_cap": 1e7},
                    {"period_end": date(2020, 6, 30), "roe": 0.25, "debt_to_equity": 0.3,
                     "net_income": 12e6, "market_cap": 1e7},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = QualityFactorStrategy(min_market_cap=5e8)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "SMALLCAP" not in result["ticker"].to_list()
