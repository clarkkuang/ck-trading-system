"""Reinforcement Learning (PPO) portfolio allocation strategy.

Uses a pre-trained Stable Baselines3 PPO model to predict optimal
portfolio weights. Train the model first with scripts/train_rl.py.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class RLStrategy(Strategy):
    """Portfolio allocation using PPO reinforcement learning."""

    def __init__(
        self,
        model_path: str = "models/rl_ppo.zip",
        config_path: str | None = None,
        tickers: list[str] | None = None,
        lookback_window: int = 252,
    ):
        self._model_path = model_path
        self._config_path = config_path or model_path.replace(".zip", ".json")
        self._tickers = tickers
        self._lookback_window = lookback_window
        self._model = None
        self._feature_builder = None

    @property
    def name(self) -> str:
        return "RL PPO"

    @property
    def description(self) -> str:
        return (
            "Reinforcement Learning portfolio allocation using PPO (Proximal Policy "
            "Optimization). Predicts optimal portfolio weights based on price momentum, "
            "volatility, RSI, and fundamental features (PE, ROE, margins). "
            "Train first: python scripts/train_rl.py --tickers ... "
            "Data: prices + fundamentals."
        )

    @property
    def rebalance_frequency(self) -> str:
        return "monthly"

    def _ensure_loaded(self, fallback_tickers: list[str]) -> None:
        """Lazy-load model and feature builder on first screen() call."""
        if self._model is not None:
            return

        from stable_baselines3 import PPO
        from ck_trading.rl.features import FeatureBuilder

        model_file = Path(self._model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"RL model not found at {self._model_path}. "
                f"Train first: python scripts/train_rl.py --tickers ..."
            )

        self._model = PPO.load(str(model_file))

        # Load ticker config if available
        config_file = Path(self._config_path)
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            self._tickers = cfg.get("tickers", self._tickers)
            self._lookback_window = cfg.get("lookback_window", self._lookback_window)

        effective_tickers = self._tickers or fallback_tickers
        self._feature_builder = FeatureBuilder(
            tickers=effective_tickers,
            lookback_window=self._lookback_window,
        )

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if prices.is_empty():
            return empty_screen_result()

        # Determine available tickers
        available = (
            prices.filter(pl.col("date") <= as_of_date)
            ["ticker"].unique().sort().to_list()
        )

        try:
            self._ensure_loaded(available)
        except FileNotFoundError as e:
            # Return empty rather than crash — UI can show a message
            return empty_screen_result()

        tickers = self._feature_builder.tickers

        # Build observation
        obs = self._feature_builder.build_observation(prices, fundamentals, as_of_date)

        # Model predicts action (deterministic at inference)
        action, _ = self._model.predict(obs, deterministic=True)

        # Softmax to get portfolio weights
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / exp_a.sum()

        # Convert to screen result
        rows = []
        for i, ticker in enumerate(tickers):
            w = float(weights[i])
            if w > 0.001:  # skip near-zero allocations
                rows.append({
                    "ticker": ticker,
                    "score": w,
                    "signal_type": "BUY",
                    "rationale": f"PPO Weight={w * 100:.1f}%",
                })

        if not rows:
            return empty_screen_result()

        return pl.DataFrame(rows).select(["ticker", "score", "signal_type", "rationale"])
