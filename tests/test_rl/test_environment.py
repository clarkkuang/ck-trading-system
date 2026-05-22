"""Tests for RL trading environment."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ck_trading.rl.environment import TradingEnv
from ck_trading.rl.features import FeatureBuilder
from ck_trading.rl.reward import SimpleReturnReward


def _make_env_data(tickers: list[str], start: date, end: date):
    """Create synthetic prices and fundamentals for environment testing."""
    import random
    rows = []
    for t in tickers:
        rng = random.Random(hash(t) % 2**31)
        price = 100.0
        d = start
        while d <= end:
            if d.weekday() < 5:
                price *= 1 + rng.gauss(0.0005, 0.015)
                rows.append({"ticker": t, "date": d, "close": price, "volume": 1_000_000,
                             "open": price, "high": price * 1.01, "low": price * 0.99})
            d += timedelta(days=1)

    prices = pl.DataFrame(rows)
    fund_rows = []
    for t in tickers:
        for q in range(1, 5):
            fund_rows.append({
                "ticker": t, "period_end": date(start.year, q * 3, 28),
                "pe_ratio": 15.0, "roe": 0.15, "market_cap": 1e9,
                "debt_to_equity": 0.5, "gross_margin": 0.4, "operating_margin": 0.15,
            })
    fundamentals = pl.DataFrame(fund_rows)
    return prices, fundamentals


class TestTradingEnv:
    def test_creation_and_spaces(self):
        tickers = ["AAPL", "MSFT"]
        prices, fund = _make_env_data(tickers, date(2020, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2020, 6, 1), end_date=date(2022, 6, 30),
        )
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (2 * 20,)

    def test_reset_returns_valid_obs(self):
        tickers = ["AAPL", "MSFT"]
        prices, fund = _make_env_data(tickers, date(2020, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2020, 6, 1), end_date=date(2022, 6, 30),
        )
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))

    def test_step_returns_correct_types(self):
        tickers = ["AAPL", "MSFT"]
        prices, fund = _make_env_data(tickers, date(2020, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2020, 6, 1), end_date=date(2022, 6, 30),
            reward_fn=SimpleReturnReward(),
        )
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "portfolio_return" in info
        assert "weights" in info

    def test_episode_terminates(self):
        tickers = ["AAPL"]
        prices, fund = _make_env_data(tickers, date(2020, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2021, 1, 1), end_date=date(2022, 12, 31),
            reward_fn=SimpleReturnReward(),
        )
        obs, _ = env.reset()
        terminated = False
        steps = 0
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if steps > 100:
                break
        assert terminated
        assert steps > 0

    def test_action_normalization(self):
        """Raw action gets softmax-normalized to sum=1."""
        tickers = ["A", "B", "C"]
        prices, fund = _make_env_data(tickers, date(2020, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2020, 6, 1), end_date=date(2022, 6, 30),
        )
        env.reset()
        action = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert pytest.approx(info["weights"].sum(), abs=0.001) == 1.0

    def test_gymnasium_check_env(self):
        """Pass gymnasium's built-in environment checker."""
        from gymnasium.utils.env_checker import check_env

        tickers = ["AAPL", "MSFT"]
        prices, fund = _make_env_data(tickers, date(2020, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2020, 6, 1), end_date=date(2022, 6, 30),
        )
        # check_env raises on failure
        check_env(env, skip_render_check=True)

    def test_walk_forward_episodes_differ(self):
        """Walk-forward mode should produce episodes with different date ranges."""
        tickers = ["AAPL"]
        prices, fund = _make_env_data(tickers, date(2018, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2018, 6, 1), end_date=date(2022, 6, 30),
            reward_fn=SimpleReturnReward(),
            walk_forward_months=12,
        )
        # Run two episodes and check that the date ranges differ (probabilistic)
        starts = set()
        for _ in range(10):
            env.reset()
            starts.add(env.rebalance_dates[0])
        # With 10 resets and ~36 possible start dates, we should get multiple
        assert len(starts) > 1, "Walk-forward should produce different episode windows"

    def test_drift_mode_reduces_turnover(self):
        """Drift mode should produce less turnover than forced rebalance."""
        tickers = ["AAPL", "MSFT"]
        prices, fund = _make_env_data(tickers, date(2020, 1, 1), date(2022, 12, 31))

        # Without drift
        env_no_drift = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2020, 6, 1), end_date=date(2022, 6, 30),
            reward_fn=SimpleReturnReward(), drift_mode=False,
        )
        env_no_drift.reset()
        action = np.array([0.7, 0.3], dtype=np.float32)
        _, _, _, _, info_nd = env_no_drift.step(action)

        # With drift (starting from equal weight, small threshold)
        env_drift = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2020, 6, 1), end_date=date(2022, 6, 30),
            reward_fn=SimpleReturnReward(), drift_mode=True, conviction_threshold=0.05,
        )
        env_drift.reset()
        # Same action, but drift mode will dampen small changes
        action_hold = np.array([0.5, 0.5], dtype=np.float32)  # "hold" = equal weight
        _, _, _, _, info_d = env_drift.step(action_hold)
        # Turnover in drift mode with "hold" action should be ~0
        assert info_d["turnover"] < info_nd["turnover"]

    def test_walk_forward_episode_length(self):
        """Walk-forward episode should be walk_forward_months + 1 dates."""
        tickers = ["AAPL"]
        prices, fund = _make_env_data(tickers, date(2018, 1, 1), date(2022, 12, 31))
        env = TradingEnv(
            prices=prices, fundamentals=fund, tickers=tickers,
            start_date=date(2018, 6, 1), end_date=date(2022, 6, 30),
            reward_fn=SimpleReturnReward(),
            walk_forward_months=12,
        )
        env.reset()
        assert len(env.rebalance_dates) == 13  # 12 months + 1
