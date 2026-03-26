"""Tests for Low Volatility Strategy."""

import random
from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.low_volatility import LowVolatilityStrategy
from ck_trading.strategies.base import empty_screen_result


def make_prices(
    ticker: str,
    start: date,
    days: int,
    start_price: float,
    daily_change: float,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic price data with controlled volatility.

    daily_change controls volatility via random.gauss(0, daily_change).
    """
    rng = random.Random(seed)
    dates = [start + timedelta(days=i) for i in range(days)]
    prices_list = [start_price]
    for _ in range(1, days):
        ret = rng.gauss(0, daily_change)
        prices_list.append(prices_list[-1] * (1 + ret))
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


EMPTY_FUND = pl.DataFrame()
AS_OF = date(2022, 1, 1)


class TestLowVolatilityStrategy:
    def test_properties(self):
        s = LowVolatilityStrategy()
        assert s.name == "Low Volatility"
        assert s.rebalance_frequency == "quarterly"
        assert "volatility" in s.description.lower()

    def test_low_vol_ranked_first(self):
        start = date(2021, 1, 1)
        a = make_prices("A", start, 300, 100.0, 0.001, seed=1)   # low vol
        b = make_prices("B", start, 300, 100.0, 0.02, seed=2)    # medium vol
        c = make_prices("C", start, 300, 100.0, 0.05, seed=3)    # high vol
        prices = pl.concat([a, b, c])

        s = LowVolatilityStrategy(lookback_days=252, top_n=3)
        result = s.screen(prices, EMPTY_FUND, AS_OF)

        assert not result.is_empty()
        tickers = result["ticker"].to_list()
        assert tickers[0] == "A"

    def test_high_vol_not_selected(self):
        start = date(2021, 1, 1)
        a = make_prices("A", start, 300, 100.0, 0.001, seed=1)
        b = make_prices("B", start, 300, 100.0, 0.02, seed=2)
        c = make_prices("C", start, 300, 100.0, 0.05, seed=3)
        prices = pl.concat([a, b, c])

        s = LowVolatilityStrategy(lookback_days=252, top_n=1)
        result = s.screen(prices, EMPTY_FUND, AS_OF)

        assert len(result) == 1
        assert result["ticker"][0] == "A"

    def test_all_buy_signals(self):
        start = date(2021, 1, 1)
        a = make_prices("A", start, 300, 100.0, 0.005, seed=1)
        b = make_prices("B", start, 300, 100.0, 0.01, seed=2)
        prices = pl.concat([a, b])

        s = LowVolatilityStrategy(top_n=10)
        result = s.screen(prices, EMPTY_FUND, AS_OF)

        assert not result.is_empty()
        signals = result["signal_type"].to_list()
        assert all(sig == "BUY" for sig in signals)

    def test_score_inversely_proportional_to_vol(self):
        start = date(2021, 1, 1)
        low = make_prices("LOW", start, 300, 100.0, 0.001, seed=10)
        high = make_prices("HIGH", start, 300, 100.0, 0.03, seed=20)
        prices = pl.concat([low, high])

        s = LowVolatilityStrategy(top_n=10)
        result = s.screen(prices, EMPTY_FUND, AS_OF)

        scores = dict(zip(result["ticker"].to_list(), result["score"].to_list()))
        assert scores["LOW"] > scores["HIGH"]

    def test_insufficient_history(self):
        start = date(2021, 11, 1)
        short = make_prices("SHORT", start, 10, 100.0, 0.01, seed=1)  # only 10 days
        long_enough = make_prices("LONG", date(2021, 1, 1), 300, 100.0, 0.01, seed=2)
        prices = pl.concat([short, long_enough])

        s = LowVolatilityStrategy(lookback_days=252)
        result = s.screen(prices, EMPTY_FUND, AS_OF)

        tickers = result["ticker"].to_list()
        assert "SHORT" not in tickers
        assert "LONG" in tickers

    def test_empty_prices(self):
        s = LowVolatilityStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUND, AS_OF)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_top_n_respected(self):
        start = date(2021, 1, 1)
        frames = []
        for i in range(5):
            frames.append(make_prices(f"T{i}", start, 300, 100.0, 0.01 * (i + 1), seed=i))
        prices = pl.concat(frames)

        s = LowVolatilityStrategy(top_n=2)
        result = s.screen(prices, EMPTY_FUND, AS_OF)

        assert len(result) == 2

    def test_rationale_contains_volatility(self):
        start = date(2021, 1, 1)
        a = make_prices("A", start, 300, 100.0, 0.01, seed=1)

        s = LowVolatilityStrategy(top_n=10)
        result = s.screen(a, EMPTY_FUND, AS_OF)

        assert not result.is_empty()
        rationale = result["rationale"][0]
        # Rationale should contain "Vol" (from "Ann. Vol=") and a numeric value
        assert "Vol" in rationale or "vol" in rationale
