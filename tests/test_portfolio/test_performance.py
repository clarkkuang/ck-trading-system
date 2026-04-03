"""Tests for portfolio performance calculation."""

from datetime import date

import polars as pl
import pytest

from ck_trading.portfolio.performance import (
    build_portfolio_values,
    calculate_drawdowns,
    calculate_returns,
    period_returns,
)

# ---------------------------------------------------------------------------
# build_portfolio_values: reconstruct daily portfolio value from trades + prices
# ---------------------------------------------------------------------------

class TestBuildPortfolioValues:
    def test_single_buy_and_hold(self):
        """Buy 100 AAPL on Jan 2, hold through Jan 5."""
        trades = [
            {"ticker": "AAPL", "action": "BUY", "shares": 100,
             "price": 150.0, "date": date(2024, 1, 2)},
        ]
        prices = pl.DataFrame({
            "ticker": ["AAPL"] * 4,
            "date": [date(2024, 1, 2), date(2024, 1, 3),
                     date(2024, 1, 4), date(2024, 1, 5)],
            "close": [150.0, 155.0, 153.0, 158.0],
        })
        result = build_portfolio_values(trades, prices, initial_capital=100_000)
        assert "date" in result.columns
        assert "portfolio_value" in result.columns
        assert result.shape[0] == 4

        values = result.sort("date")["portfolio_value"].to_list()
        # Day 1: cash 100k - 15k + 100*150 = 100k (bought at market)
        # Day 2: cash 85k + 100*155 = 100.5k
        assert values[0] == pytest.approx(100_000.0)
        assert values[1] == pytest.approx(100_500.0)

    def test_buy_then_sell(self):
        """Buy then sell some shares — cash increases, holdings decrease."""
        trades = [
            {"ticker": "AAPL", "action": "BUY", "shares": 100,
             "price": 150.0, "date": date(2024, 1, 2)},
            {"ticker": "AAPL", "action": "SELL", "shares": 50,
             "price": 160.0, "date": date(2024, 1, 4)},
        ]
        prices = pl.DataFrame({
            "ticker": ["AAPL"] * 3,
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "close": [150.0, 155.0, 160.0],
        })
        result = build_portfolio_values(trades, prices, initial_capital=100_000)
        values = result.sort("date")["portfolio_value"].to_list()
        # Day 3 (Jan 4): cash=85k+8k=93k, 50 shares * 160 = 8k => 101k
        assert values[-1] == pytest.approx(101_000.0)

    def test_multiple_tickers(self):
        """Portfolio with two stocks."""
        trades = [
            {"ticker": "AAPL", "action": "BUY", "shares": 100,
             "price": 150.0, "date": date(2024, 1, 2)},
            {"ticker": "MSFT", "action": "BUY", "shares": 50,
             "price": 370.0, "date": date(2024, 1, 2)},
        ]
        prices = pl.DataFrame({
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date": [date(2024, 1, 2), date(2024, 1, 3)] * 2,
            "close": [150.0, 160.0, 370.0, 380.0],
        })
        result = build_portfolio_values(trades, prices, initial_capital=100_000)
        values = result.sort("date")["portfolio_value"].to_list()
        # Day 1: 100k (bought at close)
        # Day 2: cash=100k-15k-18.5k=66.5k, AAPL=16k, MSFT=19k => 101.5k
        assert values[1] == pytest.approx(101_500.0)

    def test_empty_trades(self):
        result = build_portfolio_values([], pl.DataFrame(), initial_capital=100_000)
        assert result.is_empty()

    def test_no_prices_for_dates(self):
        trades = [
            {"ticker": "AAPL", "action": "BUY", "shares": 100,
             "price": 150.0, "date": date(2024, 1, 2)},
        ]
        result = build_portfolio_values(trades, pl.DataFrame(), initial_capital=100_000)
        assert result.is_empty()


# ---------------------------------------------------------------------------
# calculate_returns: daily return series from portfolio values
# ---------------------------------------------------------------------------

class TestCalculateReturns:
    def test_basic_returns(self):
        values = pl.DataFrame({
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "portfolio_value": [100_000.0, 101_000.0, 100_500.0],
        })
        result = calculate_returns(values)
        assert "daily_return" in result.columns
        rets = result["daily_return"].to_list()
        assert rets[0] == pytest.approx(0.0)  # first day no return
        assert rets[1] == pytest.approx(0.01)
        assert rets[2] == pytest.approx(-500.0 / 101_000.0)

    def test_empty_input(self):
        result = calculate_returns(pl.DataFrame())
        assert result.is_empty()


# ---------------------------------------------------------------------------
# calculate_drawdowns
# ---------------------------------------------------------------------------

class TestCalculateDrawdowns:
    def test_drawdown_series(self):
        values = pl.DataFrame({
            "date": [date(2024, 1, i) for i in range(2, 7)],
            "portfolio_value": [100.0, 110.0, 105.0, 108.0, 115.0],
        })
        result = calculate_drawdowns(values)
        assert "drawdown" in result.columns
        dd = result["drawdown"].to_list()
        # Peak at 110, drop to 105 => dd = -5/110
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(0.0)  # new peak
        assert dd[2] == pytest.approx(-5.0 / 110.0)
        assert dd[4] == pytest.approx(0.0)  # new peak at 115

    def test_monotonic_increase_no_drawdown(self):
        values = pl.DataFrame({
            "date": [date(2024, 1, i) for i in range(2, 5)],
            "portfolio_value": [100.0, 110.0, 120.0],
        })
        result = calculate_drawdowns(values)
        assert all(d == 0.0 for d in result["drawdown"].to_list())

    def test_empty_input(self):
        result = calculate_drawdowns(pl.DataFrame())
        assert result.is_empty()


# ---------------------------------------------------------------------------
# period_returns: MTD, QTD, YTD, 1Y, since inception
# ---------------------------------------------------------------------------

class TestPeriodReturns:
    @pytest.fixture
    def portfolio_values(self):
        """~1 year of daily values with simple growth."""
        dates = pl.date_range(date(2023, 1, 2), date(2024, 1, 2), eager=True)
        # Grow from 100k to ~120k linearly
        n = len(dates)
        values = [100_000 + i * (20_000 / n) for i in range(n)]
        return pl.DataFrame({"date": dates, "portfolio_value": values})

    def test_has_all_periods(self, portfolio_values):
        result = period_returns(portfolio_values, as_of=date(2024, 1, 2))
        for key in ["mtd", "qtd", "ytd", "1y", "since_inception"]:
            assert key in result

    def test_since_inception(self, portfolio_values):
        result = period_returns(portfolio_values, as_of=date(2024, 1, 2))
        # ~20% growth
        assert result["since_inception"] == pytest.approx(0.2, abs=0.01)

    def test_empty_values(self):
        result = period_returns(pl.DataFrame(), as_of=date(2024, 1, 2))
        assert result == {}

    def test_single_day(self):
        values = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "portfolio_value": [100_000.0],
        })
        result = period_returns(values, as_of=date(2024, 1, 2))
        assert result["since_inception"] == pytest.approx(0.0)
