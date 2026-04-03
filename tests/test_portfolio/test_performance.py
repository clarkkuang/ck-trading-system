"""Tests for portfolio performance calculation."""

from datetime import date

import polars as pl

from ck_trading.portfolio.performance import calculate_returns


def test_empty_trades():
    result = calculate_returns([], pl.DataFrame())
    assert result.is_empty()


def test_returns_has_expected_columns():
    trades = [
        {
            "ticker": "AAPL", "date": date(2024, 1, 15),
            "action": "BUY", "shares": 100, "price": 150.0,
        },
    ]
    prices = pl.DataFrame({
        "ticker": ["AAPL"],
        "date": [date(2024, 1, 15)],
        "close": [150.0],
    })
    result = calculate_returns(trades, prices)
    assert "date" in result.columns
    assert "portfolio_return" in result.columns
