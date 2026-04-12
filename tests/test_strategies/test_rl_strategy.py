"""Tests for RL PPO Strategy."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


def _make_prices(tickers, days=120):
    import random
    rows = []
    for t in tickers:
        rng = random.Random(hash(t))
        price = 100.0
        for i in range(days):
            d = date(2022, 1, 3) + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            price *= 1 + rng.gauss(0, 0.02)
            rows.append({"ticker": t, "date": d, "close": price,
                         "open": price, "high": price * 1.01,
                         "low": price * 0.99, "volume": 1_000_000})
    return pl.DataFrame(rows)


def _make_fundamentals(tickers):
    rows = []
    for t in tickers:
        rows.append({
            "ticker": t, "period_end": date(2022, 3, 31),
            "pe_ratio": 15.0, "roe": 0.15, "market_cap": 1e9,
            "debt_to_equity": 0.5, "gross_margin": 0.4, "operating_margin": 0.15,
        })
    return pl.DataFrame(rows)


class TestRLStrategyProperties:
    def test_name(self):
        from ck_trading.strategies.rl_strategy import RLStrategy
        s = RLStrategy()
        assert s.name == "RL PPO"

    def test_description(self):
        from ck_trading.strategies.rl_strategy import RLStrategy
        s = RLStrategy()
        assert "PPO" in s.description

    def test_rebalance_frequency(self):
        from ck_trading.strategies.rl_strategy import RLStrategy
        s = RLStrategy()
        assert s.rebalance_frequency == "monthly"


class TestRLStrategyScreen:
    def test_empty_prices_returns_empty(self):
        from ck_trading.strategies.rl_strategy import RLStrategy
        s = RLStrategy()
        result = s.screen(pl.DataFrame(), pl.DataFrame(), date(2022, 6, 1))
        assert result.is_empty()

    def test_model_not_found_returns_empty(self):
        """Missing model file → empty result (no crash)."""
        from ck_trading.strategies.rl_strategy import RLStrategy
        s = RLStrategy(model_path="nonexistent/model.zip")
        prices = _make_prices(["AAPL"])
        fund = _make_fundamentals(["AAPL"])
        result = s.screen(prices, fund, date(2022, 5, 1))
        assert result.is_empty()

    @patch("ck_trading.strategies.rl_strategy.Path")
    def test_screen_with_mock_model(self, mock_path_cls):
        """Mock PPO model → verify output DataFrame schema."""
        from ck_trading.strategies.rl_strategy import RLStrategy

        tickers = ["AAPL", "MSFT", "NVDA"]

        # Mock model that returns equal-ish weights
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.array([0.4, 0.3, 0.3], dtype=np.float32),
            None,
        )

        s = RLStrategy(tickers=tickers)
        s._model = mock_model  # inject mock

        from ck_trading.rl.features import FeatureBuilder
        s._feature_builder = FeatureBuilder(tickers=tickers)

        prices = _make_prices(tickers)
        fund = _make_fundamentals(tickers)
        result = s.screen(prices, fund, date(2022, 5, 1))

        assert not result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}
        assert all(sig == "BUY" for sig in result["signal_type"].to_list())
        # Weights should sum close to 1
        assert pytest.approx(result["score"].sum(), abs=0.05) == 1.0
        # Rationale contains PPO Weight
        assert all("PPO Weight" in r for r in result["rationale"].to_list())

    def test_lazy_loading(self):
        """Model not loaded at init time."""
        from ck_trading.strategies.rl_strategy import RLStrategy
        s = RLStrategy()
        assert s._model is None
        assert s._feature_builder is None
