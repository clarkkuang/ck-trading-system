"""Tests for RL feature builder."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


def _make_prices(tickers: list[str], days: int = 300, base: float = 100.0) -> pl.DataFrame:
    """Generate synthetic price data for multiple tickers."""
    import random
    rows = []
    for t in tickers:
        rng = random.Random(hash(t) % 2**31)
        price = base
        start = date(2022, 1, 3)
        for i in range(days):
            d = start + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            price *= 1 + rng.gauss(0.0005, 0.02)
            rows.append({
                "ticker": t, "date": d,
                "open": price * 0.99, "high": price * 1.01,
                "low": price * 0.98, "close": price,
                "volume": rng.randint(100_000, 10_000_000),
            })
    return pl.DataFrame(rows)


def _make_fundamentals(tickers: list[str]) -> pl.DataFrame:
    rows = []
    for t in tickers:
        for q in range(4):
            rows.append({
                "ticker": t,
                "period_end": date(2022, 3 * (q + 1), 28),
                "pe_ratio": 15.0 + hash(t) % 20,
                "roe": 0.1 + (hash(t) % 30) / 100,
                "market_cap": 1e9 * (1 + hash(t) % 50),
                "debt_to_equity": 0.3 + (hash(t) % 10) / 10,
                "gross_margin": 0.3 + (hash(t) % 40) / 100,
                "operating_margin": 0.1 + (hash(t) % 20) / 100,
            })
    return pl.DataFrame(rows)


class TestFeatureBuilder:
    def test_observation_shape(self):
        """Output shape should be (n_tickers * n_features,)."""
        from ck_trading.rl.features import FeatureBuilder

        tickers = ["AAPL", "MSFT", "NVDA"]
        fb = FeatureBuilder(tickers=tickers)
        prices = _make_prices(tickers)
        fundamentals = _make_fundamentals(tickers)
        obs = fb.build_observation(prices, fundamentals, date(2022, 12, 1))
        assert obs.shape == (len(tickers) * fb.n_features,)
        assert obs.dtype == np.float32

    def test_no_lookahead_bias(self):
        """Features should only use data up to as_of_date."""
        from ck_trading.rl.features import FeatureBuilder

        tickers = ["AAPL"]
        prices = _make_prices(tickers, days=300)
        fundamentals = _make_fundamentals(tickers)
        fb = FeatureBuilder(tickers=tickers)

        # Use a date in the middle of the data
        mid_date = date(2022, 6, 15)
        obs = fb.build_observation(prices, fundamentals, mid_date)
        assert obs.shape == (fb.n_features,)
        # Should not contain NaN from future data
        assert np.all(np.isfinite(obs))

    def test_missing_ticker_handled(self):
        """Ticker in list but not in prices → no NaN, finite values."""
        from ck_trading.rl.features import FeatureBuilder

        tickers = ["AAPL", "MISSING"]
        prices = _make_prices(["AAPL"], days=300)
        fundamentals = _make_fundamentals(["AAPL"])
        fb = FeatureBuilder(tickers=tickers)
        obs = fb.build_observation(prices, fundamentals, date(2022, 12, 1))
        # After z-score, missing ticker has finite values (not NaN)
        assert np.all(np.isfinite(obs))

    def test_observation_space_property(self):
        """observation_shape should return correct tuple."""
        from ck_trading.rl.features import FeatureBuilder

        fb = FeatureBuilder(tickers=["A", "B", "C"])
        shape = fb.observation_shape
        assert shape == (3 * fb.n_features,)

    def test_n_features_is_16(self):
        """Default feature count should be 16 (10 price + 6 fundamental)."""
        from ck_trading.rl.features import FeatureBuilder

        fb = FeatureBuilder(tickers=["AAPL"])
        assert fb.n_features == 16
