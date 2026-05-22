"""Portfolio-Tailored Buy the Dip Strategy.

A buy-the-dip variant tuned to a specific portfolio. Each ticker has its
own dip threshold calibrated to its historical volatility — a 5% drop in
VOO is meaningful, a 5% drop in TSLA is noise.

Unlike :class:`BuyTheDipStrategy`, this version:

* Accepts a per-ticker ``dip_thresholds`` mapping (``{ticker: min_dip_pct}``).
  Tickers outside the mapping fall back to ``default_min_dip_pct``.
* Drops the SMA-200 uptrend filter by default, so it works on bond ETFs
  and sector funds that may not be in a classic uptrend.
* Keeps the reversal-window filter to avoid catching falling knives.

The score is each ticker's dip depth normalized by its own threshold, so
two tickers triggering at 1.5× their own threshold rank equally — a
fairer cross-ticker comparison than absolute dip percent.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.buy_the_dip import compute_dip_signal
from ck_trading.strategies.registry import register


# Calibrated on 2019-04 → 2024-04 (out-of-sample vs user trade history).
# Each value is the 75th-percentile rolling 60-day drawdown for that ticker,
# clamped to [3%, 25%]. See scripts/calibrate_portfolio_dip.py for derivation.
DEFAULT_PORTFOLIO_THRESHOLDS: dict[str, float] = {
    "VOO": 0.055,
    "QQQ": 0.074,
    "INDA": 0.073,
    "MSFT": 0.085,
    "TLT": 0.082,
    "MPLX": 0.113,
    "T": 0.118,
    "NVDA": 0.127,
    "NFLX": 0.130,
    "INTC": 0.158,
    "ARKG": 0.202,
    "TSLA": 0.233,
    "RBLX": 0.250,
    # Fallbacks for related names
    "IVE": 0.060,
    "TCEHY": 0.150,
    "MSTR": 0.300,
}


@register
class PortfolioDipStrategy(Strategy):
    """Per-ticker buy-the-dip tuned to portfolio volatility profiles."""

    def __init__(
        self,
        dip_thresholds: dict[str, float] | None = None,
        default_min_dip_pct: float = 0.10,
        lookback: int = 60,
        reversal_window: int = 5,
        require_uptrend: bool = True,
        trend_period: int = 200,
        top_n: int = 10,
    ):
        self.dip_thresholds = (
            dict(DEFAULT_PORTFOLIO_THRESHOLDS)
            if dip_thresholds is None
            else dict(dip_thresholds)
        )
        self.default_min_dip_pct = default_min_dip_pct
        self.lookback = lookback
        self.reversal_window = reversal_window
        self.require_uptrend = require_uptrend
        self.trend_period = trend_period
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Portfolio Dip"

    @property
    def description(self) -> str:
        n = len(self.dip_thresholds)
        return (
            f"Per-ticker dip thresholds calibrated to historical vol "
            f"({n} tickers configured, fallback "
            f"{self.default_min_dip_pct:.0%}). "
            f"{self.lookback}-day lookback, "
            f"{self.reversal_window}-day reversal filter. Data: price only."
        )

    @property
    def rebalance_frequency(self) -> str:
        return "monthly"

    def threshold_for(self, ticker: str) -> float:
        return self.dip_thresholds.get(ticker, self.default_min_dip_pct)

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
            thr = self.threshold_for(ticker)
            signal = compute_dip_signal(
                closes,
                lookback=self.lookback,
                min_dip_pct=thr,
                trend_period=self.trend_period,
                reversal_window=self.reversal_window,
                require_uptrend=self.require_uptrend,
            )
            if signal is None:
                continue

            dip_pct = signal["dip_pct"]
            # Normalize to the ticker's own threshold so cross-ticker comparison
            # is fair (TSLA at 1.5×thr scores the same as VOO at 1.5×thr).
            score = min(dip_pct / max(thr, 1e-9), 4.0)
            rows.append({
                "ticker": ticker,
                "score": score,
                "signal_type": "BUY",
                "rationale": (
                    f"-{dip_pct:.1%} from {self.lookback}d high "
                    f"({signal['peak']:.2f} → {signal['current']:.2f}); "
                    f"threshold={thr:.1%}"
                ),
            })

        if not rows:
            return empty_screen_result()

        return (
            pl.DataFrame(rows)
            .sort("score", descending=True)
            .head(self.top_n)
            .select(["ticker", "score", "signal_type", "rationale"])
        )
