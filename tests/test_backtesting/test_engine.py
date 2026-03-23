"""Tests for the backtesting engine."""

from datetime import date

import polars as pl
import pytest

from ck_trading.backtesting.engine import BacktestEngine
from ck_trading.models.backtest import BacktestConfig
from ck_trading.strategies.graham_defensive import GrahamDefensiveStrategy


@pytest.fixture
def backtest_prices() -> pl.DataFrame:
    """Price data spanning multiple years for backtest."""
    import random

    random.seed(123)
    dates = pl.date_range(date(2019, 1, 2), date(2021, 12, 31), eager=True)
    records = []
    tickers = {"AAPL": 150, "MSFT": 100, "GOOG": 1000, "JNJ": 130, "SPY": 300}
    for ticker, base in tickers.items():
        for i, d in enumerate(dates):
            price = base + random.gauss(0, 2) + i * 0.05
            records.append({
                "ticker": ticker,
                "date": d,
                "open": price - 0.5,
                "high": price + 1.5,
                "low": price - 1.5,
                "close": round(price, 2),
                "volume": random.randint(1_000_000, 10_000_000),
                "adj_close": round(price, 2),
            })
    return pl.DataFrame(records)


@pytest.fixture
def backtest_fundamentals() -> pl.DataFrame:
    """Fundamentals that pass Graham filters."""
    base = {
        "market": "US",
        "period_end": date(2019, 12, 31),
        "report_date": date(2020, 2, 1),
        "pe_ratio": 10.0,
        "pb_ratio": 1.0,
        "current_ratio": 2.5,
        "market_cap": 5e11,
        "net_income": 20e9,
        "revenue": 100e9,
        "ebit": 25e9,
        "enterprise_value": 5.5e11,
        "total_assets": 200e9,
        "current_liabilities": 50e9,
        "cash_and_equivalents": 20e9,
        "roe": 0.25,
        "roa": 0.10,
        "operating_margin": 0.25,
        "gross_margin": 0.40,
        "debt_to_equity": 0.5,
        "total_equity": 80e9,
        "book_value_per_share": 10.0,
        "operating_cash_flow": 30e9,
        "free_cash_flow": 20e9,
    }
    rows = []
    for ticker in ["AAPL", "MSFT", "GOOG", "JNJ"]:
        row = {**base, "ticker": ticker}
        # Vary PE slightly so scoring differentiates
        row["pe_ratio"] = {"AAPL": 10, "MSFT": 11, "GOOG": 12, "JNJ": 9}[ticker]
        rows.append(row)
    return pl.DataFrame(rows)


def _make_config(**overrides) -> BacktestConfig:
    defaults = dict(
        strategy_name="Graham Defensive",
        universe=["AAPL", "MSFT", "GOOG", "JNJ"],
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
        initial_capital=100_000,
        max_positions=5,
        rebalance_freq="quarterly",
        transaction_cost_bps=10,
        benchmark="SPY",
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


class TestBacktestEngine:
    def test_basic_run_produces_results(self, backtest_prices, backtest_fundamentals):
        config = _make_config()
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=backtest_prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        assert result.config == config
        assert not result.returns.is_empty()
        assert "portfolio_return" in result.returns.columns
        assert "benchmark_return" in result.returns.columns

    def test_trades_are_generated(self, backtest_prices, backtest_fundamentals):
        config = _make_config()
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=backtest_prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        assert not result.trades.is_empty()
        assert "ticker" in result.trades.columns
        assert "action" in result.trades.columns
        # Should have BUY trades
        buy_trades = result.trades.filter(pl.col("action") == "BUY")
        assert not buy_trades.is_empty()

    def test_metrics_computed(self, backtest_prices, backtest_fundamentals):
        config = _make_config()
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=backtest_prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        assert result.metrics, "Metrics should not be empty"
        assert "total_return" in result.metrics
        assert "cagr" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "final_value" in result.metrics

    def test_final_value_changes_from_initial(self, backtest_prices, backtest_fundamentals):
        """Regression: final_value was equal to initial capital (no trades executed).

        The engine must actually produce trades with this fixture data.
        If it doesn't, the test fails — preventing silent false-green.
        """
        config = _make_config()
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=backtest_prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        # First: the engine MUST generate trades with this data
        assert not result.trades.is_empty(), (
            "Engine produced no trades — fixture data should guarantee trades"
        )
        assert result.metrics.get("total_trades", 0) > 0

        # Then: final_value must differ from initial capital
        assert result.metrics["final_value"] != config.initial_capital

    def test_max_positions_respected(self, backtest_prices, backtest_fundamentals):
        config = _make_config(max_positions=2)
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=backtest_prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        assert not result.positions.is_empty(), "Backtest should produce positions with this test data"
        for d in result.positions["date"].unique().to_list():
            day_pos = result.positions.filter(pl.col("date") == d)
            assert day_pos.height <= 2

    def test_empty_fundamentals_returns_empty_result(self, backtest_prices):
        config = _make_config()
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=backtest_prices,
            fundamentals=pl.DataFrame(),
        )
        result = engine.run()

        assert result.returns.is_empty()
        assert result.metrics == {}

    def test_no_daily_prices_in_range(self, backtest_fundamentals):
        """Prices exist only before backtest range.

        The engine's _get_prices_on_date uses '<= target_date', so stale
        historical prices CAN be used for rebalancing (trades may occur).
        But _calculate_daily_returns only uses prices within [start, end],
        so daily returns should be empty → metrics should be empty.
        """
        prices = pl.DataFrame({
            "ticker": ["AAPL"],
            "date": [date(2015, 1, 1)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000000],
            "adj_close": [100.0],
        })
        config = _make_config()
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        assert result is not None
        # No daily prices in [2020-01-01, 2020-12-31] → empty returns
        assert result.returns.is_empty(), (
            "No daily prices in backtest range → cannot compute returns"
        )
        # Metrics require non-empty returns
        assert result.metrics == {}

    def test_transaction_costs_deducted(self, backtest_prices, backtest_fundamentals):
        config = _make_config(transaction_cost_bps=100)  # 1%
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=backtest_prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        assert not result.trades.is_empty(), "Backtest should produce trades with this test data"
        assert (result.trades["cost"] > 0).all()

    def test_missing_price_day_uses_last_known(self, backtest_fundamentals):
        """Regression: missing price defaulted to 0, causing fake -100% drawdown.

        Build prices where AAPL has a gap on day 3. The engine should carry
        forward the last known price, NOT value the position at 0.
        """
        dates_list = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6),
                       date(2020, 1, 7), date(2020, 1, 8)]
        records = []
        # AAPL: present on days 1,2,4,5 — MISSING on day 3 (2020-01-06)
        aapl_prices = {
            date(2020, 1, 2): 100.0,
            date(2020, 1, 3): 101.0,
            # date(2020, 1, 6) intentionally missing
            date(2020, 1, 7): 102.0,
            date(2020, 1, 8): 103.0,
        }
        for d, p in aapl_prices.items():
            records.append({
                "ticker": "AAPL", "date": d,
                "open": p, "high": p + 1, "low": p - 1,
                "close": p, "volume": 1_000_000, "adj_close": p,
            })
        # SPY benchmark on all days
        for d in dates_list:
            records.append({
                "ticker": "SPY", "date": d,
                "open": 300.0, "high": 301.0, "low": 299.0,
                "close": 300.0, "volume": 5_000_000, "adj_close": 300.0,
            })
        prices = pl.DataFrame(records)

        config = _make_config(
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 8),
            rebalance_freq="monthly",
        )
        engine = BacktestEngine(
            strategy=GrahamDefensiveStrategy(),
            config=config,
            prices=prices,
            fundamentals=backtest_fundamentals,
        )
        result = engine.run()

        assert not result.returns.is_empty(), "Backtest should produce returns with this test data"
        rets = result.returns["portfolio_return"].to_list()
        # No daily return should be catastrophically negative
        # (old bug: missing price → 0 → return ≈ -100%)
        for r in rets:
            assert r > -0.5, (
                f"Daily return {r:.2%} is catastrophic — "
                f"likely caused by missing price defaulting to 0"
            )
