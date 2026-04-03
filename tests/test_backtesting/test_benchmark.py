"""Tests for benchmark comparison utilities."""

from datetime import date

import polars as pl

from ck_trading.backtesting.benchmark import compare_to_benchmark


def test_compare_cumulative_returns():
    portfolio = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "return": [0.01, 0.02, -0.005],
    })
    benchmark = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "return": [0.005, 0.01, 0.003],
    })
    result = compare_to_benchmark(portfolio, benchmark)
    assert "portfolio_cum" in result.columns
    assert "benchmark_cum" in result.columns
    assert "excess_cum" in result.columns
    assert result.shape[0] == 3


def test_compare_misaligned_dates():
    portfolio = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "return": [0.01, 0.02],
    })
    benchmark = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 4)],
        "return": [0.005, 0.01],
    })
    result = compare_to_benchmark(portfolio, benchmark)
    # outer join creates 3 rows: Jan 2, 3, 4
    assert result.shape[0] == 3


def test_cumulative_math():
    portfolio = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "return": [0.10, 0.10],
    })
    benchmark = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "return": [0.05, 0.05],
    })
    result = compare_to_benchmark(portfolio, benchmark)
    # Portfolio: 1.1 * 1.1 = 1.21
    # Benchmark: 1.05 * 1.05 = 1.1025
    port_cum = result["portfolio_cum"].to_list()[-1]
    bench_cum = result["benchmark_cum"].to_list()[-1]
    assert abs(port_cum - 1.21) < 0.001
    assert abs(bench_cum - 1.1025) < 0.001
