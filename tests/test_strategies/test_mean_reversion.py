"""Tests for Mean Reversion Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.mean_reversion import MeanReversionStrategy, compute_rsi


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
AS_OF = date(2022, 1, 1)


class TestMeanReversionStrategy:
    def test_properties(self):
        s = MeanReversionStrategy()
        assert s.name == "Mean Reversion"
        assert s.rebalance_frequency == "monthly"
        assert s.description

    def test_oversold_buy(self):
        """Steadily declining prices should yield RSI < 30 and a BUY signal."""
        # Create a strong downtrend
        prices = make_prices("DECL", date(2021, 11, 1), 60, 100.0, -0.02)
        s = MeanReversionStrategy(rsi_period=14, oversold_threshold=30.0)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        assert result["signal_type"][0] == "BUY"
        assert "oversold" in result["rationale"][0]

    def test_overbought_sell(self):
        """Steadily rising prices should yield RSI > 70 and a SELL signal."""
        prices = make_prices("RISE", date(2021, 11, 1), 60, 100.0, 0.02)
        s = MeanReversionStrategy(rsi_period=14, overbought_threshold=70.0)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        assert result["signal_type"][0] == "SELL"
        assert "overbought" in result["rationale"][0]

    def test_neutral_no_signal(self):
        """Oscillating prices should yield RSI ~50, no signal."""
        start = date(2021, 11, 1)
        days = 60
        dates = [start + timedelta(days=i) for i in range(days)]
        # Alternate up/down to keep RSI near 50
        prices_list = []
        p = 100.0
        for i in range(days):
            if i % 2 == 0:
                p += 1.0
            else:
                p -= 1.0
            prices_list.append(p)

        prices = pl.DataFrame({
            "ticker": ["OSC"] * days,
            "date": dates,
            "open": prices_list,
            "high": [p + 0.5 for p in prices_list],
            "low": [p - 0.5 for p in prices_list],
            "close": prices_list,
            "volume": [1_000_000] * days,
            "adj_close": prices_list,
        })

        s = MeanReversionStrategy(rsi_period=14, oversold_threshold=30.0, overbought_threshold=70.0)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        # Should be empty (no signal) because RSI is near 50
        assert result.is_empty()

    def test_rsi_calculation(self):
        """Verify RSI computation with a known 14-day sequence."""
        # Classic example: 14 days of alternating gains/losses
        # Starting at 100, then +1, -0.5 alternating
        closes = [100.0]
        for i in range(30):
            if i % 2 == 0:
                closes.append(closes[-1] + 2.0)
            else:
                closes.append(closes[-1] - 1.0)

        rsi = compute_rsi(closes, period=14)
        assert rsi is not None
        # With gains of 2 and losses of 1 alternating, RSI should be > 50
        # avg_gain > avg_loss so RSI > 50
        assert 50 < rsi < 90

    def test_empty_prices(self):
        s = MeanReversionStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_rsi_all_gains(self):
        """All-gain sequence should yield RSI = 100."""
        closes = [float(i) for i in range(100, 120)]
        rsi = compute_rsi(closes, period=14)
        assert rsi is not None
        assert rsi == 100.0

    def test_rsi_insufficient_data(self):
        """Too few data points should return None."""
        closes = [100.0, 101.0, 99.0]
        rsi = compute_rsi(closes, period=14)
        assert rsi is None
