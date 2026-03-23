"""Tests for Earnings Momentum strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.earnings_momentum import EarningsMomentumStrategy


EMPTY_PRICES = pl.DataFrame()
EMPTY_FUND = pl.DataFrame()


def test_properties():
    s = EarningsMomentumStrategy()
    assert s.name == "Earnings Momentum"
    assert s.rebalance_frequency == "monthly"
    assert s.description


def test_positive_surprise_buy():
    """Stocks with positive surprise > threshold should get BUY."""
    earnings = pl.DataFrame([
        {"ticker": "A", "report_date": date(2020, 10, 15), "surprise_pct": 0.15},
        {"ticker": "B", "report_date": date(2020, 10, 20), "surprise_pct": 0.08},
    ])
    s = EarningsMomentumStrategy(surprise_threshold=0.05, lookback_days=60)
    result = s.screen(EMPTY_PRICES, EMPTY_FUND, date(2020, 12, 1),
                      extra_data={"earnings": earnings})

    assert not result.is_empty()
    tickers = result["ticker"].to_list()
    assert "A" in tickers
    assert "B" in tickers
    assert all(st == "BUY" for st in result["signal_type"].to_list())
    # A should rank higher (higher surprise)
    scores = dict(zip(result["ticker"].to_list(), result["score"].to_list()))
    assert scores["A"] > scores["B"]


def test_negative_surprise_excluded():
    """Stocks with surprise below threshold should be excluded."""
    earnings = pl.DataFrame([
        {"ticker": "A", "report_date": date(2020, 10, 15), "surprise_pct": 0.02},  # Below 5%
        {"ticker": "B", "report_date": date(2020, 10, 20), "surprise_pct": -0.10},  # Negative
    ])
    s = EarningsMomentumStrategy(surprise_threshold=0.05, lookback_days=60)
    result = s.screen(EMPTY_PRICES, EMPTY_FUND, date(2020, 12, 1),
                      extra_data={"earnings": earnings})
    assert result.is_empty()


def test_no_extra_data_empty():
    """When extra_data is None, return empty."""
    s = EarningsMomentumStrategy()
    result = s.screen(EMPTY_PRICES, EMPTY_FUND, date(2020, 12, 31))
    assert result.is_empty()
    assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}


def test_old_earnings_excluded():
    """Earnings older than lookback_days should be excluded."""
    earnings = pl.DataFrame([
        {"ticker": "A", "report_date": date(2020, 1, 15), "surprise_pct": 0.20},  # 11 months ago
    ])
    s = EarningsMomentumStrategy(surprise_threshold=0.05, lookback_days=60)
    result = s.screen(EMPTY_PRICES, EMPTY_FUND, date(2020, 12, 1),
                      extra_data={"earnings": earnings})
    assert result.is_empty()


def test_empty_earnings():
    """Empty earnings DataFrame should return empty."""
    earnings = pl.DataFrame(schema={"ticker": pl.Utf8, "report_date": pl.Date,
                                     "surprise_pct": pl.Float64})
    s = EarningsMomentumStrategy()
    result = s.screen(EMPTY_PRICES, EMPTY_FUND, date(2020, 12, 31),
                      extra_data={"earnings": earnings})
    assert result.is_empty()
