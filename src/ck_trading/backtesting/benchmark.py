"""Benchmark comparison utilities."""

import polars as pl


def compare_to_benchmark(
    portfolio_returns: pl.DataFrame,
    benchmark_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Create a comparison DataFrame of portfolio vs benchmark cumulative returns.

    Both inputs should have columns: [date, return]
    """
    portfolio = portfolio_returns.rename({"return": "portfolio_return"})
    benchmark = benchmark_returns.rename({"return": "benchmark_return"})

    combined = portfolio.join(benchmark, on="date", how="outer_coalesce").sort("date")

    combined = combined.with_columns(
        (1 + pl.col("portfolio_return").fill_null(0)).cum_prod().alias("portfolio_cum"),
        (1 + pl.col("benchmark_return").fill_null(0)).cum_prod().alias("benchmark_cum"),
    )

    combined = combined.with_columns(
        (pl.col("portfolio_cum") - pl.col("benchmark_cum")).alias("excess_cum"),
    )

    return combined
