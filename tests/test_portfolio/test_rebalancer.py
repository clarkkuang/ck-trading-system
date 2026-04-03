"""Tests for portfolio rebalance recommender."""

import pytest

from ck_trading.portfolio.rebalancer import (
    calculate_drift,
    recommend_rebalance_trades,
)


class TestCalculateDrift:
    def test_equal_weight_no_drift(self):
        """Positions already at equal weight — zero drift."""
        current = {
            "AAPL": {"shares": 100, "market_value": 5000},
            "MSFT": {"shares": 50, "market_value": 5000},
        }
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        drift = calculate_drift(current, target_weights)
        assert drift["AAPL"] == pytest.approx(0.0)
        assert drift["MSFT"] == pytest.approx(0.0)

    def test_overweight_position(self):
        """AAPL is 70% but target is 50% — positive drift means overweight."""
        current = {
            "AAPL": {"shares": 100, "market_value": 7000},
            "MSFT": {"shares": 50, "market_value": 3000},
        }
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        drift = calculate_drift(current, target_weights)
        assert drift["AAPL"] == pytest.approx(0.2)   # 70% - 50%
        assert drift["MSFT"] == pytest.approx(-0.2)   # 30% - 50%

    def test_new_ticker_in_target(self):
        """Target has a ticker not in current holdings — drift = -target_weight."""
        current = {
            "AAPL": {"shares": 100, "market_value": 10000},
        }
        target_weights = {"AAPL": 0.5, "GOOG": 0.5}
        drift = calculate_drift(current, target_weights)
        assert drift["GOOG"] == pytest.approx(-0.5)  # need to buy
        assert drift["AAPL"] == pytest.approx(0.5)    # 100% - 50%

    def test_ticker_to_exit(self):
        """Current holds a ticker not in target — drift = current weight."""
        current = {
            "AAPL": {"shares": 100, "market_value": 5000},
            "OLD": {"shares": 10, "market_value": 5000},
        }
        target_weights = {"AAPL": 1.0}
        drift = calculate_drift(current, target_weights)
        assert drift["OLD"] == pytest.approx(0.5)  # 50% overweight, should sell
        assert drift["AAPL"] == pytest.approx(-0.5)

    def test_empty_current(self):
        drift = calculate_drift({}, {"AAPL": 0.5, "MSFT": 0.5})
        assert drift["AAPL"] == pytest.approx(-0.5)
        assert drift["MSFT"] == pytest.approx(-0.5)

    def test_empty_target(self):
        current = {"AAPL": {"shares": 100, "market_value": 10000}}
        drift = calculate_drift(current, {})
        assert drift["AAPL"] == pytest.approx(1.0)  # sell everything


class TestRecommendRebalanceTrades:
    def test_basic_rebalance(self):
        """Overweight AAPL, underweight MSFT — should sell AAPL, buy MSFT."""
        current = {
            "AAPL": {"shares": 100, "market_value": 20000},
            "MSFT": {"shares": 10, "market_value": 5000},
        }
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        prices = {"AAPL": 200.0, "MSFT": 500.0}
        trades = recommend_rebalance_trades(
            current, target_weights, prices, total_value=25000
        )
        # AAPL target = 12.5k, current = 20k → sell ~37 shares
        sell_aapl = [t for t in trades if t["ticker"] == "AAPL"]
        assert len(sell_aapl) == 1
        assert sell_aapl[0]["action"] == "SELL"
        assert sell_aapl[0]["shares"] > 0

        # MSFT target = 12.5k, current = 5k → buy ~15 shares
        buy_msft = [t for t in trades if t["ticker"] == "MSFT"]
        assert len(buy_msft) == 1
        assert buy_msft[0]["action"] == "BUY"
        assert buy_msft[0]["shares"] > 0

    def test_new_position(self):
        """Target includes a new ticker — should produce a BUY trade."""
        current = {"AAPL": {"shares": 100, "market_value": 10000}}
        target_weights = {"AAPL": 0.5, "GOOG": 0.5}
        prices = {"AAPL": 100.0, "GOOG": 150.0}
        trades = recommend_rebalance_trades(
            current, target_weights, prices, total_value=10000
        )
        buy_goog = [t for t in trades if t["ticker"] == "GOOG"]
        assert len(buy_goog) == 1
        assert buy_goog[0]["action"] == "BUY"

    def test_exit_position(self):
        """Current holds a ticker not in target — should sell all."""
        current = {
            "AAPL": {"shares": 100, "market_value": 5000},
            "OLD": {"shares": 50, "market_value": 5000},
        }
        target_weights = {"AAPL": 1.0}
        prices = {"AAPL": 50.0, "OLD": 100.0}
        trades = recommend_rebalance_trades(
            current, target_weights, prices, total_value=10000
        )
        sell_old = [t for t in trades if t["ticker"] == "OLD"]
        assert len(sell_old) == 1
        assert sell_old[0]["action"] == "SELL"
        assert sell_old[0]["shares"] == 50  # sell all

    def test_threshold_filters_small_trades(self):
        """Trades below threshold should be omitted."""
        current = {
            "AAPL": {"shares": 100, "market_value": 5100},
            "MSFT": {"shares": 50, "market_value": 4900},
        }
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        prices = {"AAPL": 51.0, "MSFT": 98.0}
        trades = recommend_rebalance_trades(
            current, target_weights, prices,
            total_value=10000, min_trade_pct=0.05,
        )
        # Drift is only 1% each way — below 5% threshold
        assert trades == []

    def test_empty_everything(self):
        trades = recommend_rebalance_trades({}, {}, {}, total_value=0)
        assert trades == []
