"""Tests for Risk Parity Strategy."""

from datetime import date, timedelta
import math

import polars as pl
import pytest

from ck_trading.strategies.risk_parity import RiskParityStrategy


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


def make_noisy_prices(
    ticker: str,
    start: date,
    days: int,
    start_price: float,
    volatility: float,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate prices with controlled volatility using a simple deterministic pattern."""
    import random
    rng = random.Random(seed)

    dates = [start + timedelta(days=i) for i in range(days)]
    prices_list = [start_price]
    for _ in range(1, days):
        ret = rng.gauss(0, volatility)
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


EMPTY_FUNDAMENTALS = pl.DataFrame()
AS_OF = date(2022, 6, 1)


class TestRiskParityStrategy:
    def test_properties(self):
        s = RiskParityStrategy()
        assert s.name == "Risk Parity"
        assert s.rebalance_frequency == "monthly"
        assert s.description

    def test_inverse_vol_weighting(self):
        """Low-vol ticker should get ~2x the weight of a ticker with 2x vol."""
        low_vol = make_noisy_prices("LOW", date(2022, 3, 1), 90, 100.0, 0.005, seed=1)
        high_vol = make_noisy_prices("HIGH", date(2022, 3, 1), 90, 100.0, 0.010, seed=2)
        prices = pl.concat([low_vol, high_vol])

        s = RiskParityStrategy(vol_lookback_days=60, min_history_days=30)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        low_row = result.filter(pl.col("ticker") == "LOW")
        high_row = result.filter(pl.col("ticker") == "HIGH")

        assert not low_row.is_empty() and not high_row.is_empty()
        low_score = low_row["score"][0]
        high_score = high_row["score"][0]
        # Lower vol => higher weight
        assert low_score > high_score

    def test_equal_vol_equal_weight(self):
        """Same volatility should give approximately equal scores."""
        a = make_noisy_prices("A", date(2022, 3, 1), 90, 100.0, 0.01, seed=10)
        b = make_noisy_prices("B", date(2022, 3, 1), 90, 100.0, 0.01, seed=10)
        # Use same seed so they have identical returns/vol
        prices = pl.concat([a, b])

        s = RiskParityStrategy(vol_lookback_days=60, min_history_days=30)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        scores = result["score"].to_list()
        assert len(scores) == 2
        assert abs(scores[0] - scores[1]) < 0.01

    def test_zero_vol_handled(self):
        """A constant-price ticker should not crash and should be excluded."""
        constant = make_prices("FLAT", date(2022, 3, 1), 90, 100.0, 0.0)
        normal = make_noisy_prices("NORM", date(2022, 3, 1), 90, 100.0, 0.01, seed=5)
        prices = pl.concat([constant, normal])

        s = RiskParityStrategy(vol_lookback_days=60, min_history_days=30)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        # Should not crash; FLAT should be excluded
        tickers = result["ticker"].to_list()
        assert "FLAT" not in tickers
        assert "NORM" in tickers

    def test_max_positions(self):
        """Only top max_positions tickers should be returned."""
        frames = []
        for i in range(10):
            frames.append(
                make_noisy_prices(f"T{i:02d}", date(2022, 3, 1), 90, 100.0, 0.005 + i * 0.002, seed=i)
            )
        prices = pl.concat(frames)

        s = RiskParityStrategy(vol_lookback_days=60, min_history_days=30, max_positions=3)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert result.height <= 3

    def test_empty_prices(self):
        s = RiskParityStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_all_buy_signals(self):
        """Risk parity should only produce BUY signals (allocation strategy)."""
        frames = []
        for i in range(5):
            frames.append(
                make_noisy_prices(f"S{i}", date(2022, 3, 1), 90, 100.0, 0.01, seed=i + 100)
            )
        prices = pl.concat(frames)

        s = RiskParityStrategy(vol_lookback_days=60, min_history_days=30)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        assert (result["signal_type"] == "BUY").all()

    def test_weights_sum_to_one(self):
        """Inverse-vol weights should sum to approximately 1.0."""
        frames = []
        for i in range(5):
            frames.append(
                make_noisy_prices(f"W{i}", date(2022, 3, 1), 90, 100.0, 0.005 + i * 0.003, seed=i + 50)
            )
        prices = pl.concat(frames)

        s = RiskParityStrategy(vol_lookback_days=60, min_history_days=30, max_positions=50)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        if not result.is_empty():
            total_weight = result["score"].sum()
            assert abs(total_weight - 1.0) < 0.01

    def test_rationale_format(self):
        """Rationale should contain Vol and Weight percentages."""
        prices = make_noisy_prices("X", date(2022, 3, 1), 90, 100.0, 0.01, seed=99)
        s = RiskParityStrategy(vol_lookback_days=60, min_history_days=30)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        rationale = result["rationale"][0]
        assert "Vol=" in rationale
        assert "Weight=" in rationale
        assert "%" in rationale
