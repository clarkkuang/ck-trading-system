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

    Default reward function. Penalizes concentration and excessive trading.
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
