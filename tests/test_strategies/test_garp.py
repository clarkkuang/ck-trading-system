"""Tests for GARP (Growth at Reasonable Price) strategy."""

from datetime import date

import polars as pl
import pytest

from ck_trading.strategies.garp import GARPStrategy


def _make_fundamentals(tickers_data: list[dict]) -> pl.DataFrame:
    """Build a multi-period fundamentals DataFrame for GARP tests.

    Each dict has:
        ticker: str
        periods: list of dicts with keys:
            period_end, net_income, pe_ratio, market_cap
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
                "net_income": p.get("net_income"),
                "pe_ratio": p.get("pe_ratio"),
                "market_cap": p.get("market_cap", 1e10),
                "revenue": 1e9,
                "roe": 0.15,
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


class TestGARPStrategy:
    def test_properties(self):
        s = GARPStrategy()
        assert s.name == "GARP"
        assert s.rebalance_frequency == "quarterly"
        assert s.description

    def test_low_peg_selected(self, empty_prices):
        """PE=10, growth=20% -> PEG=0.5 -> selected."""
        # Prior NI=100M, Latest NI=120M -> growth=20%
        # PE=10, PEG = 10/20 = 0.5
        data = [
            {
                "ticker": "LOWPEG",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 100e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 120e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = GARPStrategy(max_peg=1.5, min_growth=0.10)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert not result.is_empty()
        assert "LOWPEG" in result["ticker"].to_list()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}
        assert (result["signal_type"] == "BUY").all()

    def test_high_peg_filtered(self, empty_prices):
        """PE=30, growth=10% -> PEG=3.0 -> filtered out."""
        # Prior NI=100M, Latest NI=110M -> growth=10%
        # PE=30, PEG = 30/10 = 3.0 > max_peg=1.5
        data = [
            {
                "ticker": "HIGHPEG",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 100e6,
                     "pe_ratio": 30.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 110e6,
                     "pe_ratio": 30.0, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = GARPStrategy(max_peg=1.5, min_growth=0.10)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "HIGHPEG" not in result["ticker"].to_list()

    def test_negative_growth_excluded(self, empty_prices):
        """Declining earnings -> excluded (PEG meaningless for negative growth)."""
        data = [
            {
                "ticker": "DECLINING",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 100e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 80e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = GARPStrategy(max_peg=1.5, min_growth=0.10)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "DECLINING" not in result["ticker"].to_list()

    def test_zero_earnings_excluded(self, empty_prices):
        """Prior net_income=0 -> can't compute growth -> excluded."""
        data = [
            {
                "ticker": "ZEROBASE",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 0,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 50e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = GARPStrategy()
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "ZEROBASE" not in result["ticker"].to_list()

    def test_growth_below_minimum(self, empty_prices):
        """Growth=5% < min_growth=10% -> filtered."""
        # Prior NI=100M, Latest NI=105M -> growth=5%
        data = [
            {
                "ticker": "SLOWGROW",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 100e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 105e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = GARPStrategy(min_growth=0.10)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "SLOWGROW" not in result["ticker"].to_list()

    def test_ranking_by_peg(self, empty_prices):
        """Lower PEG ranks higher (higher score = 1/PEG)."""
        data = [
            {
                # PEG = 10/20 = 0.5 -> score = 2.0
                "ticker": "BESTPEG",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 100e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 120e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                ],
            },
            {
                # PEG = 15/20 = 0.75 -> score = 1.33
                "ticker": "MIDPEG",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 100e6,
                     "pe_ratio": 15.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 120e6,
                     "pe_ratio": 15.0, "market_cap": 5e9},
                ],
            },
            {
                # PEG = 20/15 = 1.33 -> score = 0.75
                "ticker": "WORSTPEG",
                "periods": [
                    {"period_end": date(2020, 3, 31), "net_income": 200e6,
                     "pe_ratio": 20.0, "market_cap": 5e9},
                    {"period_end": date(2020, 6, 30), "net_income": 230e6,
                     "pe_ratio": 20.0, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = GARPStrategy(max_peg=1.5, min_growth=0.10, top_n=10)
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert not result.is_empty()
        tickers = result["ticker"].to_list()
        # BESTPEG should be first (lowest PEG = highest score)
        assert tickers[0] == "BESTPEG", f"Expected BESTPEG first, got {tickers}"

        # Verify scores are descending
        scores = result["score"].to_list()
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_single_period_excluded(self, empty_prices):
        """Only 1 period -> can't compute growth -> excluded."""
        data = [
            {
                "ticker": "SINGLEPERIOD",
                "periods": [
                    {"period_end": date(2020, 6, 30), "net_income": 100e6,
                     "pe_ratio": 10.0, "market_cap": 5e9},
                ],
            },
        ]
        fundamentals = _make_fundamentals(data)
        s = GARPStrategy()
        result = s.screen(empty_prices, fundamentals, date(2020, 12, 31))

        assert result.is_empty() or "SINGLEPERIOD" not in result["ticker"].to_list()

    def test_empty_fundamentals(self, empty_prices):
        """Returns empty result for empty input."""
        s = GARPStrategy()
        result = s.screen(empty_prices, pl.DataFrame(), date(2020, 12, 31))
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}
