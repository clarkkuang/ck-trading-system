"""Tests for RL reward functions."""

import numpy as np

from ck_trading.rl.reward import RiskAdjustedReward, SharpeReward, SimpleReturnReward


class TestSimpleReturnReward:
    def test_returns_portfolio_return(self):
        r = SimpleReturnReward()
        w = np.array([0.5, 0.5])
        assert r.compute(0.05, w, w, np.array([0.03, 0.07])) == 0.05

    def test_negative_return(self):
        r = SimpleReturnReward()
        w = np.array([1.0])
        assert r.compute(-0.02, w, w, np.array([-0.02])) == -0.02


class TestSharpeReward:
    def test_positive_for_good_return(self):
        r = SharpeReward(risk_free_rate=0.0)
        w = np.array([0.5, 0.5])
        reward = r.compute(0.05, w, w, np.array([0.03, 0.07]))
        assert reward > 0

    def test_negative_for_bad_return(self):
        r = SharpeReward(risk_free_rate=0.01)
        w = np.array([0.5, 0.5])
        reward = r.compute(-0.05, w, w, np.array([-0.03, -0.07]))
        assert reward < 0


class TestRiskAdjustedReward:
    def test_turnover_reduces_reward(self):
        r = RiskAdjustedReward(risk_aversion=0.0, turnover_penalty=0.01)
        w_new = np.array([0.8, 0.2])
        w_old = np.array([0.2, 0.8])
        ret = np.array([0.05, 0.01])
        reward = r.compute(0.03, w_new, w_old, ret)
        # Without turnover penalty: reward would be 0.03
        # Turnover = |0.6| + |0.6| = 1.2, penalty = 0.012
        assert reward < 0.03

    def test_no_turnover_no_penalty(self):
        r = RiskAdjustedReward(risk_aversion=0.0, turnover_penalty=0.01)
        w = np.array([0.5, 0.5])
        reward = r.compute(0.03, w, w, np.array([0.02, 0.04]))
        # No turnover → no penalty → reward ≈ 0.03
        assert reward == pytest.approx(0.03, abs=0.001)


import pytest
