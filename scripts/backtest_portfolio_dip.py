"""Backtest the Portfolio Dip strategy against the user's holdings.

Runs four comparisons over the same window and universe:
    1. PortfolioDipStrategy (per-ticker calibrated thresholds)
    2. Generic BuyTheDipStrategy (default 10% / SMA200)
    3. Equal-weight buy-and-hold of the user's universe
    4. SPY buy-and-hold

Calibration window for thresholds (2019-04 → 2024-04) is OUT-OF-SAMPLE
with respect to the test window (2024-04 → present), so this is not
look-ahead biased.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ck_trading.backtesting.engine import BacktestEngine
from ck_trading.models.backtest import BacktestConfig
from ck_trading.strategies.buy_the_dip import BuyTheDipStrategy
from ck_trading.strategies.portfolio_dip import (
    DEFAULT_PORTFOLIO_THRESHOLDS,
    PortfolioDipStrategy,
)

REPO = Path(__file__).resolve().parent.parent
PRICES_DIR = REPO / "data" / "prices" / "us" / "daily"
DB_PATH = REPO / "data" / "metadata.db"


def user_universe() -> list[str]:
    """Distinct US tickers the user has actually traded."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM trades WHERE market = 'US'"
        ).fetchall()
    return sorted({r[0] for r in rows if not r[0].endswith(".HK")})


def load_prices(tickers: list[str]) -> pl.DataFrame:
    frames = []
    for t in tickers:
        p = PRICES_DIR / f"{t}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    return pl.concat(frames) if frames else pl.DataFrame()


def buy_and_hold_metrics(
    prices: pl.DataFrame,
    universe: list[str],
    start: date,
    end: date,
    equal_weight: bool,
) -> dict:
    """Cumulative return of equal-weight buy-and-hold of the universe."""
    sub = prices.filter(
        (pl.col("date") >= start)
        & (pl.col("date") <= end)
        & (pl.col("ticker").is_in(universe))
    )
    if sub.is_empty():
        return {"total_return": None}

    per_ticker = []
    for t in universe:
        ts = sub.filter(pl.col("ticker") == t).sort("date")
        if ts.height < 2:
            continue
        first = ts["close"][0]
        last = ts["close"][-1]
        if first > 0:
            per_ticker.append(last / first - 1)
    if not per_ticker:
        return {"total_return": None}

    if equal_weight:
        ret = sum(per_ticker) / len(per_ticker)
    else:
        ret = per_ticker[0]  # single ticker (SPY)

    days = (end - start).days or 1
    years = days / 365.25
    cagr = (1 + ret) ** (1 / years) - 1 if (1 + ret) > 0 else -1.0
    return {"total_return": ret, "cagr": cagr, "components": len(per_ticker)}


def run_strategy(strategy, prices, universe, start, end, capital, max_pos, rebal):
    config = BacktestConfig(
        strategy_name=strategy.name,
        universe=universe,
        start_date=start,
        end_date=end,
        initial_capital=capital,
        max_positions=max_pos,
        rebalance_freq=rebal,
        transaction_cost_bps=10.0,
        benchmark="SPY",
    )
    engine = BacktestEngine(
        strategy, config, prices, fundamentals=pl.DataFrame()
    )
    return engine.run()


def fmt(v, kind):
    if v is None:
        return "    n/a"
    if kind == "pct":
        return f"{v * 100:>+7.2f}%"
    if kind == "sharpe":
        # The engine reports sharpe as a fraction-of-percent
        return f"{v / 100:>7.2f}"
    return f"{v:>7.2f}"


def main() -> None:
    universe = user_universe()
    print(f"User universe ({len(universe)}): {', '.join(universe)}")

    prices = load_prices(universe + ["SPY"])
    if prices.is_empty():
        print("No price data.")
        return

    have = set(prices["ticker"].unique().to_list())
    tradable = [t for t in universe if t in have]
    if not tradable:
        print("None of the user tickers have price data.")
        return

    # Test windows: user-trade window AND a longer 5-year window
    windows = [
        ("user window (2024-04 → 2025-04)", date(2024, 4, 15), date(2025, 4, 9)),
        ("5y window  (2021-01 → 2026-03)", date(2021, 1, 1), date(2026, 3, 20)),
    ]

    for label, start, end in windows:
        print("\n" + "=" * 78)
        print(f"  {label}")
        print("=" * 78)

        # 1. Portfolio Dip (per-ticker calibrated)
        pd_strat = PortfolioDipStrategy(top_n=5)
        pd_res = run_strategy(
            pd_strat, prices, tradable, start, end, 10_000, 5, "monthly"
        )

        # 2. Generic Buy the Dip
        btd_strat = BuyTheDipStrategy(top_n=5)
        btd_res = run_strategy(
            btd_strat, prices, tradable, start, end, 10_000, 5, "monthly"
        )

        # 3. Equal-weight buy-and-hold of user's universe
        ewbh = buy_and_hold_metrics(prices, tradable, start, end, equal_weight=True)

        # 4. SPY buy-and-hold
        spybh = buy_and_hold_metrics(prices, ["SPY"], start, end, equal_weight=False)

        rows = [
            ("Portfolio Dip (per-ticker)", pd_res.metrics, pd_res.trades.height),
            ("Buy the Dip (generic 10%)", btd_res.metrics, btd_res.trades.height),
        ]
        print(
            f"  {'strategy':30}{'total':>10}{'cagr':>10}{'sharpe':>10}"
            f"{'max_dd':>10}{'alpha':>10}{'trades':>9}"
        )
        print("  " + "-" * 78)
        for name, m, ntrades in rows:
            print(
                f"  {name:30}"
                f"{fmt(m.get('total_return'), 'pct')}"
                f"{fmt(m.get('cagr'), 'pct')}"
                f"{fmt(m.get('sharpe_ratio'), 'sharpe')}"
                f"{fmt(m.get('max_drawdown'), 'pct')}"
                f"{fmt(m.get('alpha'), 'pct')}"
                f"{ntrades:>9}"
            )
        # Buy-and-hold rows (no sharpe/dd available without daily returns)
        print(
            f"  {'EW buy-and-hold (user uni.)':30}"
            f"{fmt(ewbh.get('total_return'), 'pct')}"
            f"{fmt(ewbh.get('cagr'), 'pct')}"
            f"     -      -         -"
            f"     -"
        )
        print(
            f"  {'SPY buy-and-hold':30}"
            f"{fmt(spybh.get('total_return'), 'pct')}"
            f"{fmt(spybh.get('cagr'), 'pct')}"
            f"     -      -         -"
            f"     -"
        )

        # Trades the strategy actually made
        if not pd_res.trades.is_empty():
            print("\n  Portfolio Dip trades:")
            df = pd_res.trades.sort("date")
            for row in df.iter_rows(named=True):
                print(
                    f"    {row['date']}  {row['action']:4} {row['ticker']:6} "
                    f"{row['shares']:>5}@${row['price']:>8.2f}"
                )


if __name__ == "__main__":
    print("Calibrated thresholds in use:")
    for t, v in sorted(DEFAULT_PORTFOLIO_THRESHOLDS.items(), key=lambda x: x[1]):
        print(f"  {t:6} {v:>6.1%}")
    main()
