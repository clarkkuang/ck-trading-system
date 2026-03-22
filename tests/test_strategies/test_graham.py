"""Tests for Graham Defensive strategy."""

from datetime import date

import polars as pl
import pytest

from ck_trading.strategies.graham_defensive import GrahamDefensiveStrategy


def test_graham_basic_screen(sample_prices, sample_fundamentals):
    """Test that Graham screen returns results for qualifying stocks."""
    strategy = GrahamDefensiveStrategy()
    result = strategy.screen(sample_prices, sample_fundamentals, date(2020, 12, 31))

    assert not result.is_empty()
    assert "ticker" in result.columns
    assert "score" in result.columns
    assert "signal_type" in result.columns
    assert "rationale" in result.columns

    # Both AAPL and MSFT should pass with our test data (P/E < 15, P/B < 1.5)
    tickers = result["ticker"].to_list()
    assert "MSFT" in tickers  # P/E=10, P/B=1.0 should pass


def test_graham_empty_fundamentals(sample_prices):
    """Test with empty fundamentals returns empty result."""
    strategy = GrahamDefensiveStrategy()
    result = strategy.screen(sample_prices, pl.DataFrame(), date(2020, 12, 31))
    assert result.is_empty()


def test_graham_filters_high_pe(sample_prices, sample_fundamentals):
    """Test that high P/E stocks are filtered out."""
    strategy = GrahamDefensiveStrategy(max_pe=5.0)  # Very strict
    result = strategy.screen(sample_prices, sample_fundamentals, date(2020, 12, 31))
    # All our test stocks have P/E > 5, so nothing should pass
    assert result.is_empty()


def test_graham_properties():
    """Test strategy properties."""
    strategy = GrahamDefensiveStrategy()
    assert strategy.name == "Graham Defensive"
    assert strategy.rebalance_frequency == "quarterly"
    assert strategy.description
