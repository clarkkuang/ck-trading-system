"""Dual Momentum strategy.

Combines relative and absolute momentum:
1. Relative: Rank stocks by trailing return
2. Absolute: Only keep stocks with positive returns
3. If no stocks pass, allocate to bonds (TLT)
"""

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class DualMomentumStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Dual Momentum"

    @property
    def description(self) -> str:
        return "Relative + absolute momentum with bond fallback"

    @property
    def rebalance_frequency(self) -> str:
        return "monthly"

    def __init__(
        self,
        lookback_months: int = 12,
        bond_ticker: str = "TLT",
        risk_free_rate: float = 0.02,
    ):
        self.lookback_months = lookback_months
        self.bond_ticker = bond_ticker
        self.risk_free_rate = risk_free_rate

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if prices.is_empty():
            return empty_screen_result()

        lookback_start = as_of_date - timedelta(days=self.lookback_months * 30)

        filtered = prices.filter(pl.col("date") <= as_of_date)
        if filtered.is_empty():
            return empty_screen_result()

        # Compute returns for all tickers
        returns_data = {}
        for ticker in filtered["ticker"].unique().to_list():
            ticker_df = filtered.filter(pl.col("ticker") == ticker).sort("date")

            # Get start price (near lookback_start)
            start_data = ticker_df.filter(
                (pl.col("date") >= lookback_start)
                & (pl.col("date") <= lookback_start + timedelta(days=15))
            )
            if start_data.is_empty():
                continue

            end_data = ticker_df.filter(
                pl.col("date") >= as_of_date - timedelta(days=5)
            )
            if end_data.is_empty():
                continue

            start_price = start_data.sort("date").row(0, named=True)["close"]
            end_price = end_data.sort("date", descending=True).row(0, named=True)["close"]

            if start_price > 0:
                returns_data[ticker] = (end_price / start_price) - 1.0

        if not returns_data:
            return empty_screen_result()

        bond_return = returns_data.pop(self.bond_ticker, None)

        # Relative momentum: rank stocks (excluding bond)
        stock_returns = {k: v for k, v in returns_data.items() if k != self.bond_ticker}

        if not stock_returns:
            # No stocks available, buy bonds if available
            if bond_return is not None and bond_return > 0:
                return pl.DataFrame(
                    {
                        "ticker": [self.bond_ticker],
                        "score": [bond_return],
                        "signal_type": ["BUY"],
                        "rationale": [f"No stocks available, bond return={bond_return * 100:.1f}%"],
                    }
                )
            return empty_screen_result()

        # Absolute momentum: keep only positive returns
        positive_stocks = {k: v for k, v in stock_returns.items() if v > 0}

        if not positive_stocks:
            # All stocks negative, fall back to bonds
            if bond_return is not None and bond_return > 0:
                return pl.DataFrame(
                    {
                        "ticker": [self.bond_ticker],
                        "score": [bond_return],
                        "signal_type": ["BUY"],
                        "rationale": [
                            f"All stocks negative, bond return={bond_return * 100:.1f}%"
                        ],
                    }
                )
            return empty_screen_result()

        # Build result from positive stocks, sorted by return
        sorted_stocks = sorted(positive_stocks.items(), key=lambda x: x[1], reverse=True)

        results = []
        for ticker, ret in sorted_stocks:
            results.append(
                {
                    "ticker": ticker,
                    "score": ret,
                    "signal_type": "BUY",
                    "rationale": f"12mo return={ret * 100:.1f}%, absolute momentum positive",
                }
            )

        return pl.DataFrame(results).select(["ticker", "score", "signal_type", "rationale"])
