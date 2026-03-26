"""Insider Following Strategy.

Follow corporate insider net purchases (SEC Form 4).
Large insider buying signals management confidence in the company.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class InsiderFollowingStrategy(Strategy):
    """Follow insider net buying."""

    def __init__(
        self,
        lookback_days: int = 90,
        min_buy_value: float = 50000,
        top_n: int = 20,
    ):
        self.lookback_days = lookback_days
        self.min_buy_value = min_buy_value
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Insider Following"

    @property
    def description(self) -> str:
        return (
            "Follow corporate insider net purchases (SEC Form 4). Large insider buying "
            "signals confidence in the company. Filters by lookback period and minimum "
            "buy value. Data: SEC insider trades."
        )

    @property
    def rebalance_frequency(self) -> str:
        return "monthly"

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if extra_data is None or "insider_trades" not in extra_data:
            return empty_screen_result()

        trades = extra_data["insider_trades"]
        if trades.is_empty():
            return empty_screen_result()

        required = {"ticker", "date", "transaction_type", "value"}
        if not required.issubset(set(trades.columns)):
            return empty_screen_result()

        cutoff = as_of_date - timedelta(days=self.lookback_days)

        recent = trades.filter(
            (pl.col("date") >= cutoff) & (pl.col("date") <= as_of_date)
        )

        if recent.is_empty():
            return empty_screen_result()

        # Compute net buy value per ticker: sum of P values - sum of S values
        net = (
            recent.with_columns(
                pl.when(pl.col("transaction_type") == "P")
                .then(pl.col("value"))
                .otherwise(-pl.col("value"))
                .alias("signed_value")
            )
            .group_by("ticker")
            .agg(pl.col("signed_value").sum().alias("net_buy_value"))
            .filter(pl.col("net_buy_value") > self.min_buy_value)
        )

        if net.is_empty():
            return empty_screen_result()

        net = net.with_columns(
            pl.col("net_buy_value").alias("score"),
            pl.lit("BUY").alias("signal_type"),
            (
                pl.lit("Net insider buying: $")
                + pl.col("net_buy_value").round(0).cast(pl.Utf8)
            ).alias("rationale"),
        )

        result = net.sort("score", descending=True).head(self.top_n)
        return result.select(["ticker", "score", "signal_type", "rationale"])
