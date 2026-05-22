"""Reward functions for RL trading environment."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RewardFunction(ABC):
    @abstractmethod
    def compute(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        asset_returns: np.ndarray,
    ) -> float:
        ...


class SimpleReturnReward(RewardFunction):
    """Reward = portfolio return over the step period."""

    def compute(self, portfolio_return, weights, prev_weights, asset_returns):
        return portfolio_return


class SharpeReward(RewardFunction):
    """Reward = return / volatility (step-level Sharpe proxy)."""

    def __init__(self, risk_free_rate: float = 0.04 / 252):
        self.rf = risk_free_rate

    def compute(self, portfolio_return, weights, prev_weights, asset_returns):
        excess = portfolio_return - self.rf
        # Use asset-level vol as proxy for portfolio vol
        if len(asset_returns) > 0:
            port_vol = max(np.std(asset_returns), 1e-8)
            return excess / port_vol
        return excess


class RiskAdjustedReward(RewardFunction):
    """Reward = return - risk_aversion * variance - turnover_penalty * |Δw|.

    Penalizes concentration and excessive trading.
    """

    def __init__(self, risk_aversion: float = 1.0, turnover_penalty: float = 0.001):
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty

    def compute(self, portfolio_return, weights, prev_weights, asset_returns):
        # Variance penalty from asset returns
        if len(asset_returns) > 0:
            port_var = np.var(weights * asset_returns)
        else:
            port_var = 0.0

        # Turnover penalty
        turnover = np.sum(np.abs(weights - prev_weights))

        return (
            portfolio_return
            - self.risk_aversion * port_var
            - self.turnover_penalty * turnover
        )


class BenchmarkRelativeReward(RewardFunction):
    """Reward = excess return over equal-weight benchmark.

    The model is penalized for every stock that underperforms the equal-weight
    allocation. This prevents the model from concentrating on stocks that
    happened to do well in training history but may not persist.

    reward = alpha - risk_aversion * tracking_var - turnover_penalty * |Δw|

    where alpha = portfolio_return - equal_weight_return
    """

    def __init__(
        self,
        risk_aversion: float = 0.5,
        turnover_penalty: float = 0.001,
        alpha_scale: float = 5.0,
    ):
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.alpha_scale = alpha_scale  # amplify alpha signal

    def compute(self, portfolio_return, weights, prev_weights, asset_returns):
        n = len(asset_returns)
        if n == 0:
            return 0.0

        # Equal-weight benchmark return
        eq_weight = np.ones(n) / n
        benchmark_return = float(np.sum(eq_weight * asset_returns))

        # Alpha = excess return over benchmark
        alpha = portfolio_return - benchmark_return

        # Tracking variance: variance of (w - eq_w) * returns
        active_weights = weights - eq_weight
        if len(asset_returns) > 0:
            tracking_var = np.var(active_weights * asset_returns)
        else:
            tracking_var = 0.0

        # Turnover penalty
        turnover = np.sum(np.abs(weights - prev_weights))

        return (
            self.alpha_scale * alpha
            - self.risk_aversion * tracking_var
            - self.turnover_penalty * turnover
        )


class MomentumConvictionReward(RewardFunction):
    """Reward function modeled on a "let winners run" trading style.

    Key behaviors encoded:
    1. Asymmetric turnover: selling a winner costs 3x more than other trades,
       discouraging the model from trimming positions that are performing well.
    2. Momentum holding bonus: extra reward for overweighting assets with
       positive recent returns (aligns weight with momentum).
    3. Benchmark-relative alpha: still penalizes underperforming equal weight.

    This mimics a trader who DCA-buys into positions, enables DRIP, and
    lets winning positions drift upward without rebalancing.
    """

    def __init__(
        self,
        alpha_scale: float = 5.0,
        winner_sell_penalty: float = 0.005,
        other_turnover_penalty: float = 0.001,
        momentum_bonus: float = 2.0,
    ):
        self.alpha_scale = alpha_scale
        self.winner_sell_penalty = winner_sell_penalty
        self.other_turnover_penalty = other_turnover_penalty
        self.momentum_bonus = momentum_bonus

    def compute(self, portfolio_return, weights, prev_weights, asset_returns):
        n = len(asset_returns)
        if n == 0:
            return 0.0

        # 1. Alpha over equal weight
        eq_return = float(np.mean(asset_returns))
        alpha = portfolio_return - eq_return

        # 2. Asymmetric turnover penalty
        weight_changes = weights - prev_weights
        penalty = 0.0
        for i in range(n):
            delta = abs(weight_changes[i])
            if weight_changes[i] < 0 and asset_returns[i] > 0:
                # Selling a winner → heavy penalty
                penalty += delta * self.winner_sell_penalty
            else:
                penalty += delta * self.other_turnover_penalty

        # 3. Momentum holding bonus: reward overweighting rising assets
        #    dot(weights, sign(returns)) / n — positive when heavy in winners
        momentum_alignment = float(
            np.sum(weights * np.sign(asset_returns))
        ) / n
        momentum_reward = self.momentum_bonus * momentum_alignment

        return (
            self.alpha_scale * alpha
            + momentum_reward
            - penalty
        )
