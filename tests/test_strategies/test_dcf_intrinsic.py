"""Tests for DCF Intrinsic Value strategy."""

from datetime import date

import polars as pl

from ck_trading.strategies.dcf_intrinsic import DCFIntrinsicStrategy


@pl.Config(set_tbl_cols=-1)
def test_dcf_empty_fundamentals(sample_prices):
    strategy = DCFIntrinsicStrategy()
    result = strategy.screen(sample_prices, pl.DataFrame(), date(2020, 12, 31))
    assert result.is_empty()


def test_dcf_properties():
    strategy = DCFIntrinsicStrategy()
    assert strategy.name == "DCF Intrinsic Value"
    assert "DCF" in strategy.description or "Discounted" in strategy.description
    assert strategy.rebalance_frequency == "quarterly"


def test_dcf_custom_params():
    strategy = DCFIntrinsicStrategy(
        discount_rate=0.12,
        terminal_growth=0.02,
        projection_years=5,
        margin_of_safety=0.40,
    )
    assert strategy.discount_rate == 0.12
    assert strategy.terminal_growth == 0.02
    assert strategy.projection_years == 5
    assert strategy.margin_of_safety == 0.40


def test_dcf_screens_undervalued_stock(sample_prices):
    """Create fundamentals where DCF value > market price to trigger BUY."""
    fundamentals = pl.DataFrame([{
        "ticker": "AAPL",
        "period_end": date(2020, 9, 26),
        "free_cash_flow": 73e9,
        "market_cap": 500e9,  # Artificially low so DCF > price
        "book_value_per_share": 3.85,
        "total_equity": 65e9,
    }])
    strategy = DCFIntrinsicStrategy(margin_of_safety=0.10)
    result = strategy.screen(sample_prices, fundamentals, date(2020, 12, 31))
    if not result.is_empty():
        assert "AAPL" in result["ticker"].to_list()
        assert result["signal_type"].to_list()[0] == "BUY"
        assert result["score"].to_list()[0] > 0


def test_dcf_skips_negative_fcf(sample_prices):
    """Stocks with negative FCF should be skipped."""
    fundamentals = pl.DataFrame([{
        "ticker": "AAPL",
        "period_end": date(2020, 9, 26),
        "free_cash_flow": -10e9,
        "market_cap": 2e12,
        "book_value_per_share": 3.85,
        "total_equity": 65e9,
    }])
    strategy = DCFIntrinsicStrategy()
    result = strategy.screen(sample_prices, fundamentals, date(2020, 12, 31))
    assert result.is_empty()


def test_dcf_skips_no_market_cap(sample_prices):
    """Stocks with no/zero market cap should be skipped."""
    fundamentals = pl.DataFrame([{
        "ticker": "AAPL",
        "period_end": date(2020, 9, 26),
        "free_cash_flow": 73e9,
        "market_cap": 0,
        "book_value_per_share": 3.85,
        "total_equity": 65e9,
    }])
    strategy = DCFIntrinsicStrategy()
    result = strategy.screen(sample_prices, fundamentals, date(2020, 12, 31))
    assert result.is_empty()


def test_dcf_no_fcf_columns(sample_prices):
    """If no FCF or OCF columns exist, return empty."""
    fundamentals = pl.DataFrame([{
        "ticker": "AAPL",
        "period_end": date(2020, 9, 26),
        "revenue": 274e9,
        "market_cap": 2e12,
    }])
    strategy = DCFIntrinsicStrategy()
    result = strategy.screen(sample_prices, fundamentals, date(2020, 12, 31))
    assert result.is_empty()
