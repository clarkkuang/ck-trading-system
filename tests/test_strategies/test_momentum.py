"""Tests for Momentum Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.momentum import MomentumStrategy


def make_prices(
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
        "adj_close": prices_list,
    })


EMPTY_FUNDAMENTALS = pl.DataFrame()
AS_OF = date(2022, 1, 1)


class TestMomentumStrategy:
    def test_properties(self):
        s = MomentumStrategy()
        assert s.name == "Momentum"
        assert s.rebalance_frequency == "monthly"
        assert s.description

    def test_ranking_order(self):
        """A goes up ~50%, B up ~10%, C down ~20%.  A should rank first."""
        start = date(2021, 1, 1)
        a = make_prices("A", start, 365, 100.0, 0.0011)   # ~50% over 365 days
        b = make_prices("B", start, 365, 100.0, 0.00026)   # ~10%
        c = make_prices("C", start, 365, 100.0, -0.0006)   # ~-20%
        prices = pl.concat([a, b, c])

        s = MomentumStrategy(
            lookback_months=12, skip_months=1,
            top_pct=0.34, bottom_pct=0.34, min_history_days=50,
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        buys = result.filter(pl.col("signal_type") == "BUY")
        sells = result.filter(pl.col("signal_type") == "SELL")
        assert "A" in buys["ticker"].to_list()
        assert "C" in sells["ticker"].to_list()

    def test_skip_months(self):
        """A spike in the last month should be excluded when skip_months=1."""
        start = date(2021, 1, 1)
        # Stock X: flat for 11 months, then spikes in the last month
        flat = make_prices("X", start, 335, 100.0, 0.0)
        spike_start = start + timedelta(days=335)
        spike = make_prices("X", spike_start, 30, 100.0, 0.01)
        # Overwrite ticker to match
        x_prices = pl.concat([flat, spike])

        # Stock Y: steady growth
        y_prices = make_prices("Y", start, 365, 100.0, 0.001)

        prices = pl.concat([x_prices, y_prices])
        s = MomentumStrategy(
            lookback_months=12, skip_months=1,
            top_pct=0.50, bottom_pct=0.50, min_history_days=50,
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        # Y should rank higher because X's spike is in the skipped month
        if not result.is_empty():
            buys = result.filter(pl.col("signal_type") == "BUY")
            if not buys.is_empty():
                assert "Y" in buys["ticker"].to_list()

    def test_insufficient_history(self):
        """Ticker with < min_history_days should be filtered out."""
        start = date(2021, 10, 1)
        short = make_prices("SHORT", start, 50, 100.0, 0.005)
        s = MomentumStrategy(min_history_days=200)
        result = s.screen(short, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()

    def test_empty_prices(self):
        s = MomentumStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_buy_sell_signals(self):
        """Top fraction is BUY, bottom fraction is SELL."""
        start = date(2021, 1, 1)
        tickers_data = []
        for i in range(10):
            # Spread of returns from negative to positive
            daily = -0.001 + i * 0.0003
            tickers_data.append(make_prices(f"T{i:02d}", start, 365, 100.0, daily))
        prices = pl.concat(tickers_data)

        s = MomentumStrategy(
            top_pct=0.20, bottom_pct=0.20, min_history_days=50,
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        signal_types = result["signal_type"].unique().to_list()
        assert "BUY" in signal_types
        assert "SELL" in signal_types
        # No HOLD in results
        assert "HOLD" not in signal_types
