"""Risk Parity Strategy.

Allocates capital inversely proportional to each asset's volatility,
so that every position contributes roughly equal risk.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy
from ck_trading.strategies.registry import register


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "score": pl.Float64,
            "signal_type": pl.Utf8,
            "rationale": pl.Utf8,
        }
    )


@register
class RiskParityStrategy(Strategy):
    """Inverse-volatility weighting for equal risk contribution."""

    def __init__(
        self,
        vol_lookback_days: int = 60,
        min_history_days: int = 30,
        max_positions: int = 20,
    ):
        self.vol_lookback_days = vol_lookback_days
        self.min_history_days = min_history_days
        self.max_positions = max_positions

    @property
    def name(self) -> str:
        return "Risk Parity"

    @property
    def description(self) -> str:
        return (
            f"Inverse-volatility weighting over {self.vol_lookback_days}D, "
            f"max {self.max_positions} positions"
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
        if prices.is_empty():
            return _empty_result()

        hist = prices.filter(pl.col("date") <= as_of_date)
        if hist.is_empty():
            return _empty_result()

        tickers = hist["ticker"].unique().to_list()
        ticker_vols: list[tuple[str, float]] = []

        for ticker in tickers:
            tp = hist.filter(pl.col("ticker") == ticker).sort("date")

            # Use last vol_lookback_days rows
            tp_window = tp.tail(self.vol_lookback_days)
            n = tp_window.height

            if n < self.min_history_days:
                continue

            closes = tp_window["close"].to_list()
            daily_returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
                if closes[i - 1] != 0
            ]

            if len(daily_returns) < 2:
                continue

            mean_ret = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            daily_vol = math.sqrt(variance)
            ann_vol = daily_vol * math.sqrt(252)

            # Skip zero-volatility tickers (constant prices)
            if ann_vol < 1e-10:
                continue

            ticker_vols.append((ticker, ann_vol))

        if not ticker_vols:
            return _empty_result()

        # Compute inverse-vol weights
        inv_vols = [(t, 1.0 / v) for t, v in ticker_vols]
        total_inv_vol = sum(iv for _, iv in inv_vols)

        rows: list[dict] = []
        for ticker, inv_vol in inv_vols:
            weight = inv_vol / total_inv_vol
            ann_vol = 1.0 / inv_vol  # recover original vol
            rows.append({
                "ticker": ticker,
                "score": weight,
                "signal_type": "BUY",
                "rationale": f"Vol={ann_vol * 100:.1f}% Weight={weight * 100:.1f}%",
            })

        result = pl.DataFrame(rows)

        # Top max_positions by score (= weight)
        result = result.sort("score", descending=True).head(self.max_positions)

        return result.select(["ticker", "score", "signal_type", "rationale"])
