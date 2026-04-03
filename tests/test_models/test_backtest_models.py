"""Tests for backtest models."""

from datetime import date

import polars as pl

from ck_trading.models.backtest import BacktestConfig, BacktestResult


def test_backtest_config_defaults():
    config = BacktestConfig(
        strategy_name="Graham",
        universe=["AAPL", "MSFT"],
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
    )
    assert config.initial_capital == 1_000_000
    assert config.max_positions == 20
    assert config.position_size_method == "equal_weight"
    assert config.rebalance_freq == "quarterly"
    assert config.transaction_cost_bps == 10.0
    assert config.benchmark == "SPY"


def test_backtest_config_custom():
    config = BacktestConfig(
        strategy_name="DCF",
        universe=["AAPL"],
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        initial_capital=500_000,
        max_positions=10,
        rebalance_freq="monthly",
    )
    assert config.initial_capital == 500_000
    assert config.max_positions == 10
    assert config.rebalance_freq == "monthly"


def test_backtest_result_summary():
    config = BacktestConfig(
        strategy_name="Graham",
        universe=["AAPL", "MSFT"],
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
    )
    result = BacktestResult(
        config=config,
        returns=pl.DataFrame({
            "date": [date(2020, 1, 1)],
            "portfolio_return": [0.0],
            "benchmark_return": [0.0],
        }),
        trades=pl.DataFrame(),
        positions=pl.DataFrame(),
        metrics={
            "total_return": 0.45,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15,
        },
    )
    summary = result.summary()
    assert "Graham" in summary
    assert "total_return" in summary
    assert "sharpe_ratio" in summary


def test_backtest_result_summary_with_active_dates():
    config = BacktestConfig(
        strategy_name="Test",
        universe=["AAPL"],
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
    )
    result = BacktestResult(
        config=config,
        returns=pl.DataFrame(),
        trades=pl.DataFrame(),
        positions=pl.DataFrame(),
        metrics={
            "active_start": "2020-03-15",
            "active_end": "2024-11-30",
            "total_return": 0.50,
        },
    )
    summary = result.summary()
    assert "2020-03-15" in summary
    assert "2024-11-30" in summary
