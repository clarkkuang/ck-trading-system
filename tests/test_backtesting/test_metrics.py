"""Tests for performance metrics computation."""

from datetime import date

import polars as pl
import pytest

from ck_trading.backtesting.metrics import compute_metrics


def _make_returns(portfolio_rets: list[float], benchmark_rets: list[float] | None = None) -> pl.DataFrame:
    """Helper to build a returns DataFrame."""
    n = len(portfolio_rets)
    dates = pl.date_range(date(2020, 1, 2), date(2020, 1, 2 + n - 1), eager=True)[:n]
    data = {
        "date": dates.to_list()[:n],
        "portfolio_return": portfolio_rets,
    }
    if benchmark_rets:
        data["benchmark_return"] = benchmark_rets[:n]
    else:
        data["benchmark_return"] = [0.0] * n
    return pl.DataFrame(data)


class TestComputeMetrics:
    def test_empty_returns(self):
        result = compute_metrics(pl.DataFrame())
        assert result == {}

    def test_missing_column(self):
        df = pl.DataFrame({"date": [date(2020, 1, 1)], "other": [0.01]})
        result = compute_metrics(df)
        assert result == {}

    def test_positive_returns(self):
        rets = [0.01] * 10  # 1% daily for 10 days
        df = _make_returns(rets)
        result = compute_metrics(df, initial_capital=100_000)

        assert result["total_return"] > 0
        assert result["cagr"] > 0
        assert result["final_value"] > 100_000
        assert result["max_drawdown"] == 0  # Never draws down

    def test_negative_returns(self):
        rets = [-0.01] * 10
        df = _make_returns(rets)
        result = compute_metrics(df, initial_capital=100_000)

        assert result["total_return"] < 0
        assert result["final_value"] < 100_000
        assert result["max_drawdown"] > 0

    def test_mixed_returns(self):
        rets = [0.02, -0.01, 0.03, -0.02, 0.01]
        df = _make_returns(rets)
        result = compute_metrics(df)

        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert result["annual_volatility"] > 0

    def test_win_rate(self):
        rets = [0.01, 0.02, -0.01, 0.0, 0.03]
        df = _make_returns(rets)
        result = compute_metrics(df)

        # 3 wins out of 4 non-zero days
        assert result["win_rate"] == pytest.approx(0.75)

    def test_benchmark_comparison(self):
        port_rets = [0.02] * 10
        bm_rets = [0.01] * 10
        df = _make_returns(port_rets, bm_rets)
        result = compute_metrics(df)

        assert result["alpha"] > 0  # Portfolio beats benchmark
        assert result["benchmark_total_return"] > 0

    def test_single_day(self):
        df = _make_returns([0.05])
        result = compute_metrics(df, initial_capital=100_000)

        assert result["total_return"] == pytest.approx(0.05)
        assert result["final_value"] == pytest.approx(105_000)

    def test_all_zero_returns(self):
        rets = [0.0] * 10
        df = _make_returns(rets)
        result = compute_metrics(df)

        assert result["total_return"] == pytest.approx(0.0)
        assert result["annual_volatility"] == pytest.approx(0.0)
        assert result["win_rate"] == 0.0
