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

    # Both AAPL and MSFT satisfy Graham filters:
    #   AAPL: P/E=12.5, P/B=1.2, P/E×P/B=15.0, CR=2.5, cap=$2T
    #   MSFT: P/E=10.0, P/B=1.0, P/E×P/B=10.0, CR=2.8, cap=$1.6T
    tickers = result["ticker"].to_list()
    assert "AAPL" in tickers
    assert "MSFT" in tickers

    # SPY has no fundamentals → must not appear
    assert "SPY" not in tickers


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
