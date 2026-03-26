"""Sector Rotation Strategy.

Rotate between offensive and defensive ETFs based on macro regime.
Uses yield curve (10Y-2Y spread) and VIX to detect risk-off conditions.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class SectorRotationStrategy(Strategy):
    """Rotate between offensive/defensive ETFs based on macro indicators."""

    def __init__(
        self,
        offensive_etfs: list[str] | None = None,
        defensive_etfs: list[str] | None = None,
        vix_threshold: float = 30,
    ):
        self.offensive_etfs = offensive_etfs or ["QQQ", "ARKG"]
        self.defensive_etfs = defensive_etfs or ["TLT", "VTV"]
        self.vix_threshold = vix_threshold

    @property
    def name(self) -> str:
        return "Sector Rotation"

    @property
    def description(self) -> str:
        return (
            "Rotate between offensive and defensive ETFs based on macro regime. "
            "Uses yield curve (10Y-2Y spread) and VIX to detect risk-off conditions. "
            "Data: FRED macro + price."
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
        if extra_data is None or "macro" not in extra_data:
            return empty_screen_result()

        macro = extra_data["macro"]
        if macro.is_empty() or prices.is_empty():
            return empty_screen_result()

        required = {"date", "value", "series_id"}
        if not required.issubset(set(macro.columns)):
            return empty_screen_result()

        # Get latest T10Y2Y value
        t10y2y = macro.filter(
            (pl.col("series_id") == "T10Y2Y") & (pl.col("date") <= as_of_date)
        ).sort("date", descending=True)

        # Get latest VIX value
        vix = macro.filter(
            (pl.col("series_id") == "VIXCLS") & (pl.col("date") <= as_of_date)
        ).sort("date", descending=True)

        if t10y2y.is_empty() and vix.is_empty():
            return empty_screen_result()

        yield_spread = t10y2y["value"][0] if not t10y2y.is_empty() else 0.0
        vix_value = vix["value"][0] if not vix.is_empty() else 0.0

        # Decide regime
        risk_off = (yield_spread < 0) or (vix_value > self.vix_threshold)
        selected_etfs = self.defensive_etfs if risk_off else self.offensive_etfs
        regime = "Risk-Off" if risk_off else "Risk-On"

        # Filter to ETFs that exist in prices
        available_tickers = prices["ticker"].unique().to_list()
        etfs = [e for e in selected_etfs if e in available_tickers]

        if not etfs:
            return empty_screen_result()

        # Compute 20-day momentum for selected ETFs as score
        cutoff = as_of_date - timedelta(days=30)  # buffer for 20 trading days
        window = prices.filter(
            (pl.col("ticker").is_in(etfs))
            & (pl.col("date") >= cutoff)
            & (pl.col("date") <= as_of_date)
        )

        if window.is_empty():
            return empty_screen_result()

        momentum = (
            window.sort(["ticker", "date"])
            .group_by("ticker")
            .agg(
                pl.col("close").first().alias("first_close"),
                pl.col("close").last().alias("last_close"),
            )
            .with_columns(
                ((pl.col("last_close") - pl.col("first_close")) / pl.col("first_close"))
                .alias("score")
            )
        )

        if momentum.is_empty():
            return empty_screen_result()

        momentum = momentum.with_columns(
            pl.lit("BUY").alias("signal_type"),
            (
                pl.lit(f"{regime}: Spread={yield_spread:.2f}, VIX={vix_value:.1f}; ")
                + pl.lit("20D Mom=")
                + (pl.col("score") * 100).round(1).cast(pl.Utf8)
                + pl.lit("%")
            ).alias("rationale"),
        )

        return momentum.select(["ticker", "score", "signal_type", "rationale"])
