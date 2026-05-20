"""Buy the Dip Strategy.

Buys stocks that have pulled back meaningfully from a recent rolling peak
while still being in a long-term uptrend (close above the 200-day SMA).
Optionally requires a short-term reversal confirmation (today's close
above the minimum close of the last few days) to avoid catching falling
knives.

Holdings rotate out automatically: once a name has recovered enough that
the dip is healed, it falls out of the BUY list and the backtest engine
sells it on the next rebalance.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


def compute_dip_signal(
    closes: list[float],
    lookback: int,
    min_dip_pct: float,
    trend_period: int,
    reversal_window: int,
    require_uptrend: bool,
) -> dict | None:
    """Evaluate buy-the-dip conditions on a close-price series.

    Returns a dict with ``dip_pct``, ``peak``, ``sma`` and ``current``
    when a BUY signal fires, otherwise ``None``.
    """
    needed = max(lookback, trend_period if require_uptrend else 0, reversal_window) + 1
    if len(closes) < needed:
        return None

    current = closes[-1]
    if current <= 0:
        return None

    window = closes[-lookback:]
    peak = max(window)
    if peak <= 0:
        return None
    dip_pct = (peak - current) / peak
    if dip_pct < min_dip_pct:
        return None

    if require_uptrend:
        trend_window = closes[-trend_period:]
        sma = sum(trend_window) / len(trend_window)
        if current < sma:
            return None
    else:
        sma = current

    if reversal_window > 1:
        recent_low = min(closes[-reversal_window:])
        # Require today's close to be above the recent low (not still falling)
        if current <= recent_low:
            return None

    return {"dip_pct": dip_pct, "peak": peak, "sma": sma, "current": current}


@register
class BuyTheDipStrategy(Strategy):
    """Buy meaningful pullbacks within an uptrend."""

    def __init__(
        self,
        lookback: int = 60,
        min_dip_pct: float = 0.10,
        trend_period: int = 200,
        reversal_window: int = 5,
        require_uptrend: bool = True,
        top_n: int = 20,
    ):
        self.lookback = lookback
        self.min_dip_pct = min_dip_pct
        self.trend_period = trend_period
        self.reversal_window = reversal_window
        self.require_uptrend = require_uptrend
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Buy the Dip"

    @property
    def description(self) -> str:
        trend = f", above SMA{self.trend_period}" if self.require_uptrend else ""
        return (
            f"Buy when price is >={self.min_dip_pct:.0%} below its "
            f"{self.lookback}-day rolling high{trend}, with a "
            f"{self.reversal_window}-day reversal filter. Data: price only."
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
            return empty_screen_result()

        hist = prices.filter(pl.col("date") <= as_of_date)
        if hist.is_empty():
            return empty_screen_result()

        rows: list[dict] = []
        for ticker in hist["ticker"].unique().to_list():
            closes = (
                hist.filter(pl.col("ticker") == ticker)
                .sort("date")["close"]
                .to_list()
            )
            signal = compute_dip_signal(
                closes,
                lookback=self.lookback,
                min_dip_pct=self.min_dip_pct,
                trend_period=self.trend_period,
                reversal_window=self.reversal_window,
                require_uptrend=self.require_uptrend,
            )
            if signal is None:
                continue

            dip_pct = signal["dip_pct"]
            # Score grows with dip depth; cap so a 40% dip doesn't dominate
            score = min(dip_pct / max(self.min_dip_pct, 1e-9), 4.0)
            rows.append({
                "ticker": ticker,
                "score": score,
                "signal_type": "BUY",
                "rationale": (
                    f"-{dip_pct:.1%} from {self.lookback}d high "
                    f"({signal['peak']:.2f} → {signal['current']:.2f})"
                ),
            })

        if not rows:
            return empty_screen_result()

        result = (
            pl.DataFrame(rows)
            .sort("score", descending=True)
            .head(self.top_n)
            .select(["ticker", "score", "signal_type", "rationale"])
        )
        return result
