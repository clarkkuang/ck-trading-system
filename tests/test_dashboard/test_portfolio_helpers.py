"""Tests for portfolio page helper functions (TDD — RED first)."""

from datetime import date

import pytest


class TestPositionsToTrades:
    def test_positions_to_trades_basic(self):
        """Convert a position dict to the trade format build_portfolio_values expects."""
        from ck_trading.dashboard.portfolio_helpers import positions_to_trades

        positions = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0, "date_acquired": date(2023, 1, 15), "market": "us"},
            {"ticker": "MSFT", "shares": 5, "avg_cost": 300.0, "date_acquired": date(2023, 6, 1), "market": "us"},
        ]
        trades = positions_to_trades(positions)
        assert len(trades) == 2
        assert trades[0]["ticker"] == "AAPL"
        assert trades[0]["action"] == "BUY"
        assert trades[0]["shares"] == 10
        assert trades[0]["price"] == 150.0
        assert trades[0]["date"] == date(2023, 1, 15)

    def test_positions_to_trades_empty(self):
        """Empty positions list returns empty trades list."""
        from ck_trading.dashboard.portfolio_helpers import positions_to_trades

        assert positions_to_trades([]) == []


class TestBuildCurrentHoldings:
    def test_build_current_holdings(self):
        """Build {ticker: {shares, market_value}} from positions + latest prices."""
        from ck_trading.dashboard.portfolio_helpers import build_current_holdings

        positions = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
            {"ticker": "MSFT", "shares": 5, "avg_cost": 300.0},
        ]
        latest_prices = {"AAPL": 170.0, "MSFT": 350.0}
        result = build_current_holdings(positions, latest_prices)

        assert result["AAPL"]["shares"] == 10
        assert result["AAPL"]["market_value"] == 1700.0
        assert result["MSFT"]["shares"] == 5
        assert result["MSFT"]["market_value"] == 1750.0

    def test_build_current_holdings_missing_price(self):
        """Ticker not in prices → market_value uses avg_cost as fallback."""
        from ck_trading.dashboard.portfolio_helpers import build_current_holdings

        positions = [{"ticker": "UNKNOWN", "shares": 10, "avg_cost": 50.0}]
        result = build_current_holdings(positions, {})
        assert result["UNKNOWN"]["market_value"] == 500.0  # shares * avg_cost fallback


class TestBuildRiskPositions:
    def test_build_risk_positions(self):
        """Enrich positions with market_value and sector from universe."""
        from ck_trading.dashboard.portfolio_helpers import build_risk_positions

        positions = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
        ]
        latest_prices = {"AAPL": 170.0}
        universe = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "JPM", "sector": "Financials"},
        ]
        result = build_risk_positions(positions, latest_prices, universe)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["market_value"] == 1700.0
        assert result[0]["sector"] == "Technology"

    def test_build_risk_positions_unknown_sector(self):
        """Ticker not in universe → sector is 'Unknown'."""
        from ck_trading.dashboard.portfolio_helpers import build_risk_positions

        positions = [{"ticker": "XYZ", "shares": 5, "avg_cost": 20.0}]
        result = build_risk_positions(positions, {"XYZ": 25.0}, [])
        assert result[0]["sector"] == "Unknown"


class TestPositionsToTradesStringDate:
    def test_string_date_converted(self):
        """String date_acquired should be converted to date object."""
        from ck_trading.dashboard.portfolio_helpers import positions_to_trades

        positions = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0,
             "date_acquired": "2023-01-15", "market": "us"},
        ]
        trades = positions_to_trades(positions)
        assert trades[0]["date"] == date(2023, 1, 15)
        assert isinstance(trades[0]["date"], date)


class TestBuildPortfolioValueSeries:
    def test_basic(self):
        """Compute daily portfolio value from positions + prices."""
        import polars as pl
        from ck_trading.dashboard.portfolio_helpers import build_portfolio_value_series

        positions = [
            {"ticker": "AAPL", "shares": 10},
            {"ticker": "MSFT", "shares": 5},
        ]
        prices = pl.DataFrame({
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "date": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)] * 2,
            "close": [150.0, 152.0, 155.0, 300.0, 305.0, 310.0],
        })
        result = build_portfolio_value_series(positions, prices)
        assert not result.is_empty()
        assert "date" in result.columns
        assert "portfolio_value" in result.columns
        # Day 1: 10*150 + 5*300 = 3000
        assert result.filter(pl.col("date") == date(2023, 1, 2))["portfolio_value"][0] == 3000.0
        # Day 3: 10*155 + 5*310 = 3100
        assert result.filter(pl.col("date") == date(2023, 1, 4))["portfolio_value"][0] == 3100.0

    def test_empty_positions(self):
        """Empty positions → empty result."""
        import polars as pl
        from ck_trading.dashboard.portfolio_helpers import build_portfolio_value_series

        prices = pl.DataFrame({"ticker": ["AAPL"], "date": [date(2023, 1, 2)], "close": [150.0]})
        result = build_portfolio_value_series([], prices)
        assert result.is_empty()

    def test_ticker_not_in_prices(self):
        """Position ticker not in prices → that ticker contributes 0."""
        import polars as pl
        from ck_trading.dashboard.portfolio_helpers import build_portfolio_value_series

        positions = [{"ticker": "XYZ", "shares": 10}]
        prices = pl.DataFrame({"ticker": ["AAPL"], "date": [date(2023, 1, 2)], "close": [150.0]})
        result = build_portfolio_value_series(positions, prices)
        assert result.is_empty()  # no price data for XYZ → no rows
