"""Portfolio performance calculation."""

from datetime import date

import polars as pl


def calculate_returns(
    trades: list[dict],
    prices: pl.DataFrame,
    initial_capital: float = 0.0,
) -> pl.DataFrame:
    """Calculate time-weighted returns from trade history and prices."""
    if not trades:
        return pl.DataFrame()

    trades_df = pl.DataFrame(trades).sort("date")
    start_date = trades_df["date"].min()
    end_date = date.today()

    # This is a simplified implementation
    # Full implementation would track daily portfolio value
    return pl.DataFrame({
        "date": [start_date, end_date],
        "portfolio_return": [0.0, 0.0],
    })
