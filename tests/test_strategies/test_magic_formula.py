"""Tests for Magic Formula strategy."""

from datetime import date

import polars as pl
import pytest

from ck_trading.strategies.magic_formula import MagicFormulaStrategy


class TestMagicFormulaStrategy:
    def test_properties(self):
        s = MagicFormulaStrategy()
        assert s.name == "Magic Formula"
        assert s.rebalance_frequency == "annual"
        assert s.description

    def test_basic_screen(self, sample_prices, sample_fundamentals):
        s = MagicFormulaStrategy(top_n=10, min_market_cap=1e8)
        result = s.screen(sample_prices, sample_fundamentals, date(2020, 12, 31))

        assert not result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}
        assert (result["signal_type"] == "BUY").all()

    def test_empty_fundamentals(self, sample_prices):
        s = MagicFormulaStrategy()
        result = s.screen(sample_prices, pl.DataFrame(), date(2020, 12, 31))
        assert result.is_empty()

    def test_ranking_order(self, sample_prices, sample_fundamentals):
        s = MagicFormulaStrategy(top_n=10)
        result = s.screen(sample_prices, sample_fundamentals, date(2020, 12, 31))

        if result.height > 1:
            scores = result["score"].to_list()
            # Top scores should be sorted descending by combined rank
            assert scores == sorted(scores, reverse=True)

    def test_market_cap_filter(self, sample_prices):
        """Stocks below min_market_cap are excluded."""
        fundamentals = pl.DataFrame([
            {
                "ticker": "SMALL",
                "market": "US",
                "period_end": date(2020, 6, 30),
                "pe_ratio": 8.0,
                "ebit": 1e6,
                "enterprise_value": 5e6,
                "total_assets": 10e6,
                "current_liabilities": 2e6,
                "cash_and_equivalents": 1e6,
                "market_cap": 50_000,  # Very small
            },
        ])
        s = MagicFormulaStrategy(min_market_cap=1e8)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))
        assert result.is_empty()

    def test_pe_fallback(self, sample_prices):
        """Uses 1/PE when EBIT/EV not available."""
        fundamentals = pl.DataFrame([
            {
                "ticker": "AAPL",
                "market": "US",
                "period_end": date(2020, 6, 30),
                "pe_ratio": 10.0,
                "roe": 0.25,
                "market_cap": 2e12,
            },
            {
                "ticker": "MSFT",
                "market": "US",
                "period_end": date(2020, 6, 30),
                "pe_ratio": 15.0,
                "roe": 0.30,
                "market_cap": 1.5e12,
            },
        ])
        s = MagicFormulaStrategy(top_n=5)
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))
        # Should still produce results using PE fallback
        assert not result.is_empty()

    def test_negative_ebit_filtered(self, sample_prices):
        """Companies with negative EBIT are excluded."""
        fundamentals = pl.DataFrame([
            {
                "ticker": "LOSS",
                "market": "US",
                "period_end": date(2020, 6, 30),
                "ebit": -5e6,
                "enterprise_value": 1e8,
                "total_assets": 50e6,
                "current_liabilities": 10e6,
                "market_cap": 1e9,
            },
        ])
        s = MagicFormulaStrategy()
        result = s.screen(sample_prices, fundamentals, date(2020, 12, 31))
        assert result.is_empty()

    def test_rationale_contains_metrics(self, sample_prices, sample_fundamentals):
        s = MagicFormulaStrategy(top_n=10)
        result = s.screen(sample_prices, sample_fundamentals, date(2020, 12, 31))

        if not result.is_empty():
            rationale = result["rationale"][0]
            assert "EY=" in rationale
            assert "ROC=" in rationale
