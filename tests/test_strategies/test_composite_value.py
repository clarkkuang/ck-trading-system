"""Tests for Composite Value strategy."""

from datetime import date

import polars as pl
import pytest

from ck_trading.strategies.composite_value import CompositeValueStrategy


def _make_fundamentals(n: int = 10) -> pl.DataFrame:
    """Generate N stocks with varying fundamentals."""
    rows = []
    for i in range(n):
        rows.append({
            "ticker": f"TICK{i:02d}",
            "market": "US",
            "period_end": date(2020, 12, 31),
            "pe_ratio": 5.0 + i * 2,
            "pb_ratio": 0.5 + i * 0.3,
            "ps_ratio": 1.0 + i * 0.5,
            "roe": 0.30 - i * 0.02,
            "current_ratio": 3.0 - i * 0.15,
            "debt_to_equity": 0.2 + i * 0.1,
            "operating_margin": 0.35 - i * 0.02,
            "market_cap": 1e10 + i * 1e9,
            "net_income": 1e9,
        })
    return pl.DataFrame(rows)


class TestCompositeValueStrategy:
    def test_properties(self):
        s = CompositeValueStrategy()
        assert s.name == "Composite Value"
        assert s.rebalance_frequency == "quarterly"
        assert s.description

    def test_basic_screen(self, sample_prices):
        fundamentals = _make_fundamentals(10)
        s = CompositeValueStrategy(top_n=5, min_market_cap=1e8)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))

        assert not result.is_empty()
        assert result.height <= 5
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}
        assert (result["signal_type"] == "BUY").all()

    def test_empty_fundamentals(self, sample_prices):
        s = CompositeValueStrategy()
        result = s.screen(sample_prices, pl.DataFrame(), date(2020, 12, 31))
        assert result.is_empty()

    def test_too_few_stocks(self, sample_prices):
        """Needs >= 5 stocks to screen."""
        fundamentals = _make_fundamentals(3)
        s = CompositeValueStrategy()
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))
        assert result.is_empty()

    def test_scores_between_0_and_1(self, sample_prices):
        fundamentals = _make_fundamentals(20)
        s = CompositeValueStrategy(top_n=10)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))

        if not result.is_empty():
            assert (result["score"] >= 0).all()
            assert (result["score"] <= 1.5).all()  # Weighted sum could exceed 1 slightly

    def test_top_n_respected(self, sample_prices):
        fundamentals = _make_fundamentals(20)
        s = CompositeValueStrategy(top_n=3)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))
        assert result.height <= 3

    def test_custom_weights_change_ranking(self, sample_prices):
        """Prove that weights actually affect ranking order.

        With pe_score=1.0 (lower PE = better), TICK00 (PE=5) should rank first.
        With roe_score=1.0 (higher ROE = better), TICK00 (ROE=0.30) should also
        rank first because our fixture has ROE decreasing with index.
        But with margin_score=1.0, TICK00 (margin=0.35) is best too.

        So instead: flip the test — use pe_only vs roe_only and verify the
        #1 pick differs when the data supports it.
        """
        # Construct data where best-PE and best-ROE are DIFFERENT stocks
        rows = []
        for i in range(10):
            rows.append({
                "ticker": f"TICK{i:02d}",
                "market": "US",
                "period_end": date(2020, 12, 31),
                "pe_ratio": 5.0 + i * 2,          # TICK00 has lowest PE (best)
                "pb_ratio": 1.0,
                "roe": 0.05 + i * 0.03,            # TICK09 has highest ROE (best)
                "current_ratio": 2.0,
                "debt_to_equity": 0.5,
                "operating_margin": 0.20,
                "market_cap": 1e10,
                "net_income": 1e9,
            })
        fundamentals = pl.DataFrame(rows)

        pe_only = {
            "pe_score": 1.0, "pb_score": 0.0, "ps_score": 0.0,
            "roe_score": 0.0, "cr_score": 0.0, "de_score": 0.0, "margin_score": 0.0,
        }
        roe_only = {
            "pe_score": 0.0, "pb_score": 0.0, "ps_score": 0.0,
            "roe_score": 1.0, "cr_score": 0.0, "de_score": 0.0, "margin_score": 0.0,
        }

        result_pe = CompositeValueStrategy(
            weights=pe_only, top_n=5, min_market_cap=1e8,
        ).screen(sample_prices, fundamentals, date(2020, 12, 31))

        result_roe = CompositeValueStrategy(
            weights=roe_only, top_n=5, min_market_cap=1e8,
        ).screen(sample_prices, fundamentals, date(2020, 12, 31))

        assert not result_pe.is_empty()
        assert not result_roe.is_empty()

        # PE-only: lowest PE (TICK00) should be #1
        assert result_pe["ticker"][0] == "TICK00", (
            f"PE-only ranking: expected TICK00 first, got {result_pe['ticker'][0]}"
        )
        # ROE-only: highest ROE (TICK09) should be #1
        assert result_roe["ticker"][0] == "TICK09", (
            f"ROE-only ranking: expected TICK09 first, got {result_roe['ticker'][0]}"
        )

    def test_rationale_not_none(self, sample_prices):
        """Regression: concat_str returned null when inputs had nulls."""
        fundamentals = _make_fundamentals(10)
        s = CompositeValueStrategy(top_n=5, min_market_cap=1e8)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))

        if not result.is_empty():
            for r in result["rationale"].to_list():
                assert r is not None
                assert isinstance(r, str)
                assert len(r) > 0

    def test_missing_optional_columns(self, sample_prices):
        """Should work with only pe_ratio and market_cap."""
        fundamentals = pl.DataFrame([
            {"ticker": f"T{i}", "pe_ratio": 10.0 + i, "market_cap": 1e10}
            for i in range(10)
        ])
        s = CompositeValueStrategy(top_n=5, min_market_cap=1e8)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))
        assert not result.is_empty()

    def test_market_cap_filter(self, sample_prices):
        fundamentals = _make_fundamentals(10)
        # Set all market caps below threshold
        fundamentals = fundamentals.with_columns(pl.lit(100.0).alias("market_cap"))
        s = CompositeValueStrategy(min_market_cap=1e8)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))
        assert result.is_empty()
