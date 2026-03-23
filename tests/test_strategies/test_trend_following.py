"""Tests for Trend Following Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.trend_following import TrendFollowingStrategy


def make_prices(
    ticker: str,
    start: date,
    days: int,
    start_price: float,
    daily_change: float,
) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(days)]
    prices_list = [start_price * (1 + daily_change) ** i for i in range(days)]
    return pl.DataFrame({
        "ticker": [ticker] * days,
        "date": dates,
        "open": prices_list,
        "high": [p * 1.01 for p in prices_list],
        "low": [p * 0.99 for p in prices_list],
        "close": prices_list,
        "volume": [1_000_000] * days,
        "adj_close": prices_list,
    })


EMPTY_FUNDAMENTALS = pl.DataFrame()
AS_OF = date(2022, 6, 1)


class TestTrendFollowingStrategy:
    def test_properties(self):
        s = TrendFollowingStrategy()
        assert s.name == "Trend Following"
        assert s.rebalance_frequency == "monthly"
        assert s.description

    def test_uptrend_buy(self):
        """250 days of steadily rising prices should produce a BUY signal."""
        prices = make_prices("UP", date(2021, 6, 1), 350, 100.0, 0.002)
        s = TrendFollowingStrategy(fast_sma=50, slow_sma=200)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        assert result["signal_type"][0] == "BUY"
        assert result["ticker"][0] == "UP"

    def test_downtrend_sell(self):
        """250 days of declining prices should produce a SELL signal."""
        prices = make_prices("DOWN", date(2021, 6, 1), 350, 200.0, -0.002)
        s = TrendFollowingStrategy(fast_sma=50, slow_sma=200)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        assert result["signal_type"][0] == "SELL"
        assert result["ticker"][0] == "DOWN"

    def test_insufficient_history(self):
        """Fewer than slow_sma days of data should produce no signal."""
        prices = make_prices("SHORT", date(2022, 1, 1), 150, 100.0, 0.002)
        s = TrendFollowingStrategy(slow_sma=200)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()

    def test_atr_in_rationale(self):
        """ATR and StopLoss values should appear in the rationale string."""
        prices = make_prices("UP", date(2021, 6, 1), 350, 100.0, 0.002)
        s = TrendFollowingStrategy()
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        rationale = result["rationale"][0]
        assert "ATR=" in rationale
        assert "StopLoss=" in rationale
        assert "SMA50=" in rationale
        assert "SMA200=" in rationale

    def test_empty_prices(self):
        s = TrendFollowingStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_score_positive(self):
        """Score should be positive for both BUY and SELL signals."""
        up = make_prices("UP", date(2021, 6, 1), 350, 100.0, 0.002)
        down = make_prices("DOWN", date(2021, 6, 1), 350, 200.0, -0.002)
        prices = pl.concat([up, down])

        s = TrendFollowingStrategy()
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        assert (result["score"] > 0).all()
