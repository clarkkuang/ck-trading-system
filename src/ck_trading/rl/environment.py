"""Gymnasium trading environment for RL portfolio optimization."""

from __future__ import annotations

from datetime import date, timedelta

import gymnasium
import numpy as np
import polars as pl

from ck_trading.rl.features import FeatureBuilder
from ck_trading.rl.reward import RewardFunction, RiskAdjustedReward


def _generate_monthly_dates(start: date, end: date) -> list[date]:
    """Generate month-end rebalance dates between start and end."""
    import calendar
    dates = []
    y, m = start.year, start.month
    while True:
        last_day = calendar.monthrange(y, m)[1]
        d = date(y, m, last_day)
        if d > end:
            break
        if d >= start:
            dates.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


class TradingEnv(gymnasium.Env):
    """Portfolio allocation environment.

    At each step the agent outputs target weights for N tickers.
    The environment simulates holding that portfolio until the next
    rebalance date, computes returns, deducts transaction costs,
    and returns the reward.

    Walk-forward mode (walk_forward_months > 0):
      Each episode uses a random sub-window of the training period.
      This exposes the agent to different market regimes and prevents
      overfitting to one specific start-to-end trajectory.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        tickers: list[str],
        start_date: date,
        end_date: date,
        feature_builder: FeatureBuilder | None = None,
        reward_fn: RewardFunction | None = None,
        transaction_cost_bps: float = 10.0,
        walk_forward_months: int = 0,
        drift_mode: bool = False,
        conviction_threshold: float = 0.02,
    ):
        super().__init__()

        self.prices = prices
        self.fundamentals = fundamentals
        self.tickers = list(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.feature_builder = feature_builder or FeatureBuilder(tickers=self.tickers)
        self.reward_fn = reward_fn or RiskAdjustedReward()
        self.tc_rate = transaction_cost_bps / 10_000
        self.walk_forward_months = walk_forward_months
        self.drift_mode = drift_mode
        self.conviction_threshold = conviction_threshold

        # Precompute all rebalance dates in the full window
        self._all_rebalance_dates = _generate_monthly_dates(start_date, end_date)
        if len(self._all_rebalance_dates) < 2:
            raise ValueError("Need at least 2 rebalance dates")

        # Active rebalance dates for current episode (may be subset if walk-forward)
        self.rebalance_dates = self._all_rebalance_dates

        n_tickers = len(self.tickers)
        n_features = self.feature_builder.n_features

        self.action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(n_tickers,), dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_tickers * n_features,), dtype=np.float32,
        )

        # Precompute daily close prices per ticker for fast return lookups
        self._price_map = self._build_price_map()

        # State
        self._step_idx = 0
        self._weights = np.ones(n_tickers, dtype=np.float32) / n_tickers

    def _build_price_map(self) -> dict[str, dict[date, float]]:
        """Build {ticker: {date: close}} for fast lookups."""
        pmap: dict[str, dict[date, float]] = {t: {} for t in self.tickers}
        if self.prices.is_empty():
            return pmap
        for row in self.prices.filter(
            pl.col("ticker").is_in(self.tickers)
        ).iter_rows(named=True):
            pmap[row["ticker"]][row["date"]] = row["close"]
        return pmap

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Walk-forward: pick a random sub-window for this episode
        if self.walk_forward_months > 0 and len(self._all_rebalance_dates) > self.walk_forward_months + 1:
            max_start = len(self._all_rebalance_dates) - self.walk_forward_months - 1
            start_idx = self.np_random.integers(0, max_start + 1)
            end_idx = start_idx + self.walk_forward_months + 1
            self.rebalance_dates = self._all_rebalance_dates[start_idx:end_idx]
        else:
            self.rebalance_dates = self._all_rebalance_dates

        self._step_idx = 0
        self._weights = np.ones(len(self.tickers), dtype=np.float32) / len(self.tickers)
        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        # Normalize action to target weights via softmax
        exp_a = np.exp(action - np.max(action))
        target_weights = (exp_a / exp_a.sum()).astype(np.float32)

        # Current and next rebalance dates
        current_date = self.rebalance_dates[self._step_idx]
        self._step_idx += 1
        terminated = self._step_idx >= len(self.rebalance_dates) - 1
        next_date = self.rebalance_dates[min(self._step_idx, len(self.rebalance_dates) - 1)]

        # Compute asset returns between current_date and next_date
        asset_returns = self._compute_period_returns(current_date, next_date)

        if self.drift_mode:
            # Natural drift: only trade the delta between drifted weights and target
            # First, compute where current weights would drift to based on returns
            drifted = self._weights * (1 + asset_returns)
            drift_sum = drifted.sum()
            if drift_sum > 0:
                drifted_weights = (drifted / drift_sum).astype(np.float32)
            else:
                drifted_weights = self._weights.copy()

            # Apply conviction threshold: only trade if delta > threshold
            delta = target_weights - drifted_weights
            # Zero out small adjustments (model learns to "hold")
            delta = np.where(np.abs(delta) > self.conviction_threshold, delta, 0.0)
            new_weights = (drifted_weights + delta).astype(np.float32)
            # Re-normalize
            w_sum = new_weights.sum()
            if w_sum > 0:
                new_weights = new_weights / w_sum
        else:
            new_weights = target_weights

        # Portfolio return (weighted sum of asset returns)
        portfolio_return = float(np.sum(new_weights * asset_returns))

        # Transaction costs
        turnover = float(np.sum(np.abs(new_weights - self._weights)))
        tc = turnover * self.tc_rate
        portfolio_return -= tc

        # Reward
        reward = float(self.reward_fn.compute(
            portfolio_return, new_weights, self._weights, asset_returns,
        ))

        # Update state
        self._weights = new_weights

        # Next observation
        obs = self._get_observation()

        info = {
            "date": next_date,
            "portfolio_return": portfolio_return,
            "weights": new_weights.copy(),
            "turnover": turnover,
        }

        return obs, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        idx = min(self._step_idx, len(self.rebalance_dates) - 1)
        as_of = self.rebalance_dates[idx]
        return self.feature_builder.build_observation(
            self.prices, self.fundamentals, as_of,
        )

    def _compute_period_returns(self, start: date, end: date) -> np.ndarray:
        """Compute per-ticker returns between two dates."""
        returns = np.zeros(len(self.tickers), dtype=np.float32)
        for i, ticker in enumerate(self.tickers):
            pm = self._price_map.get(ticker, {})
            # Find closest prices to start and end dates
            start_price = self._find_nearest_price(pm, start)
            end_price = self._find_nearest_price(pm, end)
            if start_price and start_price > 0 and end_price:
                returns[i] = end_price / start_price - 1
        return returns

    @staticmethod
    def _find_nearest_price(
        price_dict: dict[date, float], target: date, max_lookback: int = 10,
    ) -> float | None:
        """Find price on target date or nearest prior date."""
        for offset in range(max_lookback):
            d = target - timedelta(days=offset)
            if d in price_dict:
                return price_dict[d]
        return None
