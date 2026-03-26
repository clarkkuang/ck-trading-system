"""Earnings Surprise Strategy.

Buy stocks with consecutive earnings beats (Post-Earnings Announcement Drift).
Targets companies that consistently exceed analyst estimates.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class EarningsSurpriseStrategy(Strategy):
    """Consecutive earnings beats -> PEAD effect."""

    def __init__(
        self,
        min_consecutive_beats: int = 2,
        min_surprise_pct: float = 0.03,
        top_n: int = 20,
    ):
        self.min_consecutive_beats = min_consecutive_beats
        self.min_surprise_pct = min_surprise_pct
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Earnings Surprise"

    @property
    def description(self) -> str:
        return (
            "Buy stocks with consecutive earnings beats (Post-Earnings Announcement "
            "Drift). Targets companies that consistently exceed analyst estimates. "
            "Data: earnings history."
        )

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if extra_data is None or "earnings" not in extra_data:
            return empty_screen_result()

        earnings = extra_data["earnings"]
        if earnings.is_empty():
            return empty_screen_result()

        # Support both 'date' and 'report_date' column names
        date_col = None
        for col_name in ("report_date", "date"):
            if col_name in earnings.columns:
                date_col = col_name
                break

        if date_col is None or "ticker" not in earnings.columns or "surprise_pct" not in earnings.columns:
            return empty_screen_result()

        # Filter to earnings on or before as_of_date
        past = earnings.filter(
            (pl.col(date_col) <= as_of_date)
            & pl.col("surprise_pct").is_not_null()
        )

        if past.is_empty():
            return empty_screen_result()

        # For each ticker, sort by date desc and count consecutive beats
        results = []
        for ticker in past["ticker"].unique().to_list():
            ticker_data = (
                past.filter(pl.col("ticker") == ticker)
                .sort(date_col, descending=True)
            )

            surprises = ticker_data["surprise_pct"].to_list()
            consecutive = 0
            total_surprise = 0.0
            for s in surprises:
                if s >= self.min_surprise_pct:
                    consecutive += 1
                    total_surprise += s
                else:
                    break

            if consecutive >= self.min_consecutive_beats:
                avg_surprise = total_surprise / consecutive
                results.append({
                    "ticker": ticker,
                    "score": avg_surprise,
                    "signal_type": "BUY",
                    "rationale": (
                        f"{consecutive} consecutive beats, "
                        f"avg surprise={avg_surprise * 100:.1f}%"
                    ),
                })

        if not results:
            return empty_screen_result()

        result_df = pl.DataFrame(results).sort("score", descending=True).head(self.top_n)
        return result_df.select(["ticker", "score", "signal_type", "rationale"])
