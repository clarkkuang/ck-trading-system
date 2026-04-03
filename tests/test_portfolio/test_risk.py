"""Tests for portfolio risk metrics."""

import pytest

from ck_trading.portfolio.risk import concentration_analysis


def test_empty_positions():
    result = concentration_analysis([])
    assert result["hhi"] == 0.0
    assert result["top5_weight"] == 0.0
    assert result["by_market"] == {}


def test_single_position():
    positions = [{"ticker": "AAPL", "market": "US", "sector": "Tech", "market_value": 10000}]
    result = concentration_analysis(positions)
    assert result["hhi"] == pytest.approx(1.0)
    assert result["top5_weight"] == pytest.approx(1.0)
    assert result["by_market"]["US"] == pytest.approx(1.0)
    assert result["num_positions"] == 1


def test_equal_weight_two_positions():
    positions = [
        {"ticker": "AAPL", "market": "US", "sector": "Tech", "market_value": 5000},
        {"ticker": "MSFT", "market": "US", "sector": "Tech", "market_value": 5000},
    ]
    result = concentration_analysis(positions)
    assert result["hhi"] == pytest.approx(0.5)
    assert result["top5_weight"] == pytest.approx(1.0)


def test_mixed_markets():
    positions = [
        {"ticker": "AAPL", "market": "US", "sector": "Tech", "market_value": 7000},
        {"ticker": "0700.HK", "market": "HK", "sector": "Tech", "market_value": 3000},
    ]
    result = concentration_analysis(positions)
    assert result["by_market"]["US"] == pytest.approx(0.7)
    assert result["by_market"]["HK"] == pytest.approx(0.3)


def test_by_sector():
    positions = [
        {"ticker": "AAPL", "market": "US", "sector": "Tech", "market_value": 6000},
        {"ticker": "JPM", "market": "US", "sector": "Finance", "market_value": 4000},
    ]
    result = concentration_analysis(positions)
    assert result["by_sector"]["Tech"] == pytest.approx(0.6)
    assert result["by_sector"]["Finance"] == pytest.approx(0.4)


def test_zero_total_value():
    positions = [{"ticker": "AAPL", "market": "US", "market_value": 0}]
    result = concentration_analysis(positions)
    assert result["hhi"] == 0.0


def test_top5_weight_with_many_positions():
    # 10 equal positions
    positions = [
        {"ticker": f"T{i}", "market": "US", "sector": "Tech", "market_value": 1000}
        for i in range(10)
    ]
    result = concentration_analysis(positions)
    assert result["top5_weight"] == pytest.approx(0.5)
    assert result["num_positions"] == 10
