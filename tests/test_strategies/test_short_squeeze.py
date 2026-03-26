"""Tests for Short Squeeze Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.short_squeeze import ShortSqueezeStrategy
from ck_trading.strategies.base import empty_screen_result


EMPTY_FUND = pl.DataFrame()
AS_OF = date(2022, 1, 1)


def make_prices_trend(
    ticker: str,
    start: date,
    days: int,
    start_price: float,
    daily_change: float,
) -> pl.DataFrame:
    """Generate synthetic price data with a constant daily growth rate."""
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
    })


def make_short_interest(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(records).with_columns(pl.col("date").cast(pl.Date))


class TestShortSqueezeStrategy:
    def test_properties(self):
        s = ShortSqueezeStrategy()
        assert s.name == "Short Squeeze"
        assert s.rebalance_frequency == "monthly"
        assert s.description

    def test_high_short_plus_momentum(self):
        si = make_short_interest([
            {"ticker": "GME", "date": date(2021, 12, 28), "short_ratio": 0.25},
        ])
        # Trending up ~10% over 10 days
        prices = make_prices_trend("GME", date(2021, 12, 20), 12, 100.0, 0.008)

        s = ShortSqueezeStrategy(min_short_ratio=0.15, min_momentum_pct=0.05,
                                  lookback_days=5)
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"short_interest": si})

        assert not result.is_empty()
        assert "GME" in result["ticker"].to_list()

    def test_high_short_no_momentum(self):
        si = make_short_interest([
            {"ticker": "BAD", "date": date(2021, 12, 28), "short_ratio": 0.25},
        ])
        # Trending down
        prices = make_prices_trend("BAD", date(2021, 12, 20), 12, 100.0, -0.01)

        s = ShortSqueezeStrategy(min_short_ratio=0.15, min_momentum_pct=0.05,
                                  lookback_days=5)
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"short_interest": si})

        assert result.is_empty()

    def test_low_short_excluded(self):
        si = make_short_interest([
            {"ticker": "SAFE", "date": date(2021, 12, 28), "short_ratio": 0.05},
        ])
        prices = make_prices_trend("SAFE", date(2021, 12, 20), 12, 100.0, 0.01)

        s = ShortSqueezeStrategy(min_short_ratio=0.15, min_momentum_pct=0.05)
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"short_interest": si})

        assert result.is_empty()

    def test_no_extra_data_returns_empty(self):
        s = ShortSqueezeStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUND, AS_OF, extra_data=None)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_score_calculation(self):
        si = make_short_interest([
            {"ticker": "X", "date": date(2021, 12, 28), "short_ratio": 0.30},
        ])
        # Create strong uptrend for clear momentum
        prices = make_prices_trend("X", date(2021, 12, 20), 12, 100.0, 0.01)

        s = ShortSqueezeStrategy(min_short_ratio=0.15, min_momentum_pct=0.01,
                                  lookback_days=5)
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"short_interest": si})

        assert not result.is_empty()
        score = result["score"][0]
        # Score should be short_ratio * momentum, both positive
        assert score > 0

    def test_empty_prices_returns_empty(self):
        si = make_short_interest([
            {"ticker": "X", "date": date(2021, 12, 28), "short_ratio": 0.30},
        ])
        s = ShortSqueezeStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUND, AS_OF,
                          extra_data={"short_interest": si})
        assert result.is_empty()

    def test_top_n_respected(self):
        records = []
        frames = []
        for i in range(5):
            ticker = f"T{i}"
            records.append({"ticker": ticker, "date": date(2021, 12, 28),
                            "short_ratio": 0.20 + i * 0.05})
            frames.append(make_prices_trend(ticker, date(2021, 12, 20), 12,
                                             100.0, 0.01))
        si = make_short_interest(records)
        prices = pl.concat(frames)

        s = ShortSqueezeStrategy(min_short_ratio=0.15, min_momentum_pct=0.01,
                                  lookback_days=5, top_n=2)
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"short_interest": si})

        assert len(result) <= 2
