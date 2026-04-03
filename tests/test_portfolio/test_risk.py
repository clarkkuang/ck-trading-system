"""Tests for portfolio risk metrics."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.portfolio.risk import (
    concentration_analysis,
    correlation_matrix,
    portfolio_var,
    stress_test,
)

# ---- Existing concentration tests ----

class TestConcentrationAnalysis:
    def test_empty_positions(self):
        result = concentration_analysis([])
        assert result["hhi"] == 0.0
        assert result["top5_weight"] == 0.0
        assert result["by_market"] == {}

    def test_single_position(self):
        positions = [
            {"ticker": "AAPL", "market": "US", "sector": "Tech",
             "market_value": 10000},
        ]
        result = concentration_analysis(positions)
        assert result["hhi"] == pytest.approx(1.0)
        assert result["top5_weight"] == pytest.approx(1.0)
        assert result["by_market"]["US"] == pytest.approx(1.0)
        assert result["num_positions"] == 1

    def test_equal_weight_two_positions(self):
        positions = [
            {"ticker": "AAPL", "market": "US", "sector": "Tech",
             "market_value": 5000},
            {"ticker": "MSFT", "market": "US", "sector": "Tech",
             "market_value": 5000},
        ]
        result = concentration_analysis(positions)
        assert result["hhi"] == pytest.approx(0.5)
        assert result["top5_weight"] == pytest.approx(1.0)

    def test_mixed_markets(self):
        positions = [
            {"ticker": "AAPL", "market": "US", "sector": "Tech",
             "market_value": 7000},
            {"ticker": "0700.HK", "market": "HK", "sector": "Tech",
             "market_value": 3000},
        ]
        result = concentration_analysis(positions)
        assert result["by_market"]["US"] == pytest.approx(0.7)
        assert result["by_market"]["HK"] == pytest.approx(0.3)

    def test_by_sector(self):
        positions = [
            {"ticker": "AAPL", "market": "US", "sector": "Tech",
             "market_value": 6000},
            {"ticker": "JPM", "market": "US", "sector": "Finance",
             "market_value": 4000},
        ]
        result = concentration_analysis(positions)
        assert result["by_sector"]["Tech"] == pytest.approx(0.6)
        assert result["by_sector"]["Finance"] == pytest.approx(0.4)

    def test_zero_total_value(self):
        positions = [{"ticker": "AAPL", "market": "US", "market_value": 0}]
        result = concentration_analysis(positions)
        assert result["hhi"] == 0.0

    def test_top5_weight_with_many_positions(self):
        positions = [
            {"ticker": f"T{i}", "market": "US", "sector": "Tech",
             "market_value": 1000}
            for i in range(10)
        ]
        result = concentration_analysis(positions)
        assert result["top5_weight"] == pytest.approx(0.5)
        assert result["num_positions"] == 10


# ---- Correlation matrix ----

def _make_price_series(ticker: str, base: float, n: int = 60):
    """Helper: deterministic daily prices."""
    import random
    random.seed(hash(ticker) % 2**31)
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]
    prices_list = []
    price = base
    for d in dates:
        price *= 1 + random.gauss(0, 0.01)
        prices_list.append(
            {"ticker": ticker, "date": d, "close": price}
        )
    return prices_list


class TestCorrelationMatrix:
    def test_shape(self):
        rows = _make_price_series("AAPL", 150) + _make_price_series("MSFT", 370)
        prices = pl.DataFrame(rows)
        corr = correlation_matrix(prices, ["AAPL", "MSFT"])
        assert corr.shape == (2, 2)

    def test_diagonal_is_one(self):
        rows = _make_price_series("AAPL", 150) + _make_price_series("MSFT", 370)
        prices = pl.DataFrame(rows)
        corr = correlation_matrix(prices, ["AAPL", "MSFT"])
        assert corr["AAPL"][0] == pytest.approx(1.0)  # row 0 = AAPL
        assert corr["MSFT"][1] == pytest.approx(1.0)  # row 1 = MSFT

    def test_symmetry(self):
        rows = (
            _make_price_series("AAPL", 150)
            + _make_price_series("MSFT", 370)
            + _make_price_series("GOOG", 140)
        )
        prices = pl.DataFrame(rows)
        corr = correlation_matrix(prices, ["AAPL", "MSFT", "GOOG"])
        # Row 0=AAPL, Row 1=MSFT: corr[MSFT][0] == corr[AAPL][1]
        assert corr["MSFT"][0] == pytest.approx(corr["AAPL"][1])

    def test_single_ticker(self):
        rows = _make_price_series("AAPL", 150)
        prices = pl.DataFrame(rows)
        corr = correlation_matrix(prices, ["AAPL"])
        assert corr.shape == (1, 1)
        assert corr["AAPL"][0] == pytest.approx(1.0)

    def test_empty_prices(self):
        corr = correlation_matrix(pl.DataFrame(), ["AAPL"])
        assert corr.shape == (0, 0)


# ---- Stress testing ----

class TestStressTest:
    @pytest.fixture
    def positions(self):
        return [
            {"ticker": "AAPL", "market_value": 10000, "sector": "Tech"},
            {"ticker": "MSFT", "market_value": 8000, "sector": "Tech"},
            {"ticker": "JPM", "market_value": 7000, "sector": "Finance"},
        ]

    def test_uniform_shock(self, positions):
        """Apply -20% to all positions."""
        result = stress_test(
            positions, scenarios={"Market Crash": {"all": -0.20}}
        )
        assert "Market Crash" in result
        assert result["Market Crash"]["portfolio_impact"] == pytest.approx(-0.20)
        assert result["Market Crash"]["dollar_impact"] == pytest.approx(-5000.0)

    def test_sector_shock(self, positions):
        """Tech drops 30%, Finance flat."""
        result = stress_test(
            positions,
            scenarios={"Tech Crash": {"Tech": -0.30, "Finance": 0.0}},
        )
        r = result["Tech Crash"]
        # Tech = 18k, 30% loss = -5.4k. Finance flat. Total = 25k
        assert r["dollar_impact"] == pytest.approx(-5400.0)
        expected_pct = -5400.0 / 25000.0
        assert r["portfolio_impact"] == pytest.approx(expected_pct)

    def test_ticker_specific_shock(self, positions):
        """Individual ticker shock overrides sector."""
        result = stress_test(
            positions,
            scenarios={"AAPL Drop": {"AAPL": -0.50}},
        )
        r = result["AAPL Drop"]
        assert r["dollar_impact"] == pytest.approx(-5000.0)

    def test_multiple_scenarios(self, positions):
        result = stress_test(
            positions,
            scenarios={
                "Mild": {"all": -0.05},
                "Severe": {"all": -0.40},
            },
        )
        assert len(result) == 2
        assert abs(result["Severe"]["dollar_impact"]) > abs(
            result["Mild"]["dollar_impact"]
        )

    def test_empty_positions(self):
        result = stress_test([], scenarios={"Crash": {"all": -0.20}})
        assert result["Crash"]["dollar_impact"] == 0.0
        assert result["Crash"]["portfolio_impact"] == 0.0


# ---- Portfolio VaR ----

class TestPortfolioVaR:
    def test_basic_var(self):
        """VaR from historical return series."""
        rows = _make_price_series("AAPL", 150, n=252)
        prices = pl.DataFrame(rows)
        var = portfolio_var(
            prices,
            holdings={"AAPL": 10000.0},
            confidence=0.95,
        )
        # VaR should be negative (loss)
        assert var < 0
        # And reasonable — not more than the entire holding
        assert var > -10000.0

    def test_higher_confidence_means_larger_var(self):
        rows = _make_price_series("AAPL", 150, n=252)
        prices = pl.DataFrame(rows)
        var_95 = portfolio_var(
            prices, holdings={"AAPL": 10000.0}, confidence=0.95
        )
        var_99 = portfolio_var(
            prices, holdings={"AAPL": 10000.0}, confidence=0.99
        )
        # 99% VaR should be more extreme (more negative) than 95%
        assert var_99 < var_95

    def test_empty_holdings(self):
        prices = pl.DataFrame()
        var = portfolio_var(prices, holdings={}, confidence=0.95)
        assert var == 0.0
