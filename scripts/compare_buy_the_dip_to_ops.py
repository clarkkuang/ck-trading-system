"""Compare Buy-the-Dip signals to your actual BUY operations.

For every distinct date you placed a non-DRIP BUY trade, generate the
strategy's signal set as-of that date and check whether the ticker you
bought was on it (same-day) or within ±5 calendar days.

This complements scripts/backtest_buy_the_dip.py (which only backtests
the strategy) with a behavioral question: how often did your trades
agree with what the strategy would have flagged?
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ck_trading.strategies.buy_the_dip import BuyTheDipStrategy

REPO = Path(__file__).resolve().parent.parent
PRICES_DIR = REPO / "data" / "prices" / "us" / "daily"
DB_PATH = REPO / "data" / "metadata.db"


def load_user_buys() -> pl.DataFrame:
    """Active BUYs from the trades table (excluding DRIP)."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            """
            SELECT date, ticker
            FROM trades
            WHERE action = 'BUY' AND (notes IS NULL OR notes != 'DRIP')
            ORDER BY date
            """
        ).fetchall()
    return pl.DataFrame({
        "date": [date.fromisoformat(r[0]) for r in rows],
        "ticker": [r[1] for r in rows],
    })


def load_prices(tickers: list[str]) -> pl.DataFrame:
    frames = []
    for t in tickers:
        path = PRICES_DIR / f"{t}.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
    return pl.concat(frames) if frames else pl.DataFrame()


def main() -> None:
    buys = load_user_buys()
    if buys.is_empty():
        print("No BUY trades found.")
        return

    universe = sorted(set(buys["ticker"].to_list()))
    prices = load_prices(universe)
    if prices.is_empty():
        print("No price data for any user ticker.")
        return

    have = set(prices["ticker"].unique().to_list())
    missing = [t for t in universe if t not in have]
    if missing:
        print(f"Note: missing price data for {missing}")

    strategy = BuyTheDipStrategy(top_n=50)
    empty_fund = pl.DataFrame()

    # Cache signals per as-of date so we don't recompute for ±5d window
    signal_cache: dict[date, set[str]] = {}

    def signals_on(d: date) -> set[str]:
        if d not in signal_cache:
            r = strategy.screen(prices, empty_fund, d)
            signal_cache[d] = set(r["ticker"].to_list()) if not r.is_empty() else set()
        return signal_cache[d]

    per_ticker = defaultdict(lambda: {"total": 0, "exact": 0, "near": 0})
    total = exact = near = 0

    distinct_dates = sorted(set(buys["date"].to_list()))
    by_date_tickers = {
        row["date"]: set(row["ticker"])
        for row in buys.group_by("date").agg(pl.col("ticker")).iter_rows(named=True)
    }

    for as_of in distinct_dates:
        same_day = signals_on(as_of)
        for ticker in by_date_tickers.get(as_of, set()):
            total += 1
            per_ticker[ticker]["total"] += 1
            on_today = ticker in same_day
            if on_today:
                exact += 1
                near += 1
                per_ticker[ticker]["exact"] += 1
                per_ticker[ticker]["near"] += 1
                continue
            for offset in range(-5, 6):
                if offset == 0:
                    continue
                if ticker in signals_on(as_of + timedelta(days=offset)):
                    near += 1
                    per_ticker[ticker]["near"] += 1
                    break

    pct = lambda x: (x / total * 100) if total else 0
    print(f"User non-DRIP buys evaluated: {total}")
    print(f"Window: {distinct_dates[0]} → {distinct_dates[-1]}")
    print(f"  Exact same-day match: {exact} ({pct(exact):.1f}%)")
    print(f"  Within ±5 days:       {near} ({pct(near):.1f}%)")

    print(f"\n  {'ticker':<8}{'#buys':>7}{'exact':>8}{'±5d':>8}")
    for t in sorted(per_ticker, key=lambda x: per_ticker[x]["total"], reverse=True):
        s = per_ticker[t]
        print(f"  {t:<8}{s['total']:>7}{s['exact']:>8}{s['near']:>8}")


if __name__ == "__main__":
    main()
