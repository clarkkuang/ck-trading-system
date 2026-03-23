"""Tests for Dual Momentum strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.dual_momentum import DualMomentumStrategy


def make_prices(ticker, start, days, start_price, daily_return=0.0):
    dates = [start + timedelta(days=i) for i in range(days)]
    prices = [start_price * (1 + daily_return) ** i for i in range(days)]
    return pl.DataFrame({
        "ticker": [ticker] * days, "date": dates,
        "open": prices, "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices], "close": prices,
        "volume": [1_000_000] * days, "adj_close": prices,
    })


EMPTY_FUND = pl.DataFrame()


def test_properties():
    s = DualMomentumStrategy()
    assert s.name == "Dual Momentum"
    assert s.rebalance_frequency == "monthly"
    assert s.description


def test_stocks_beat_bonds():
    """When stocks have positive returns beating bonds, buy stocks."""
    start = date(2019, 1, 1)
    days = 400
    a = make_prices("A", start, days, 100.0, 0.002)    # ~+80% annually
    b = make_prices("B", start, days, 100.0, 0.001)    # ~+40% annually
    tlt = make_prices("TLT", start, days, 100.0, 0.0002)  # ~8% annually

    prices = pl.concat([a, b, tlt])
    s = DualMomentumStrategy(lookback_months=12, bond_ticker="TLT")
    result = s.screen(prices, EMPTY_FUND, start + timedelta(days=days - 1))

    assert not result.is_empty()
    tickers = result["ticker"].to_list()
    assert "A" in tickers
    assert "TLT" not in tickers  # Stocks beat bonds
    assert all(st == "BUY" for st in result["signal_type"].to_list())


def test_bonds_beat_stocks():
    """When all stocks are negative but bonds are positive, buy TLT."""
    start = date(2019, 1, 1)
    days = 400
    a = make_prices("A", start, days, 100.0, -0.001)    # negative
    b = make_prices("B", start, days, 100.0, -0.002)    # negative
    tlt = make_prices("TLT", start, days, 100.0, 0.0003)  # positive

    prices = pl.concat([a, b, tlt])
    s = DualMomentumStrategy(lookback_months=12, bond_ticker="TLT")
    result = s.screen(prices, EMPTY_FUND, start + timedelta(days=days - 1))

    assert not result.is_empty()
    assert result["ticker"][0] == "TLT"
    assert result["signal_type"][0] == "BUY"


def test_all_negative():
    """When stocks and bonds are all negative, return empty or no BUY."""
    start = date(2019, 1, 1)
    days = 400
    a = make_prices("A", start, days, 100.0, -0.001)
    tlt = make_prices("TLT", start, days, 100.0, -0.0005)

    prices = pl.concat([a, tlt])
    s = DualMomentumStrategy(lookback_months=12, bond_ticker="TLT")
    result = s.screen(prices, EMPTY_FUND, start + timedelta(days=days - 1))

    # Should be empty (all stocks negative, bond also negative)
    assert result.is_empty(), "Strategy should produce no results when all returns are negative"


def test_no_tlt_graceful():
    """When TLT is not in prices, strategy should still work for stocks."""
    start = date(2019, 1, 1)
    days = 400
    a = make_prices("A", start, days, 100.0, 0.002)
    b = make_prices("B", start, days, 100.0, 0.001)

    prices = pl.concat([a, b])
    s = DualMomentumStrategy(lookback_months=12, bond_ticker="TLT")
    result = s.screen(prices, EMPTY_FUND, start + timedelta(days=days - 1))

    assert not result.is_empty()
    assert "TLT" not in result["ticker"].to_list()


def test_empty_prices():
    s = DualMomentumStrategy()
    result = s.screen(pl.DataFrame(), EMPTY_FUND, date(2020, 12, 31))
    assert result.is_empty()
    assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}
