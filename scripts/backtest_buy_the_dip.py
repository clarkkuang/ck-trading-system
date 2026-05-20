"""Backtest the Buy the Dip strategy against your portfolio.

Universe priority:
  1. Tickers passed via --tickers TICKER1,TICKER2,...
  2. Open positions in the SQLite metadata store (your real portfolio)
  3. A default sample portfolio of large-cap US names

If price data is not present in the local Parquet store, it is downloaded
on the fly via yfinance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ck_trading.backtesting.engine import BacktestEngine
from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.models.backtest import BacktestConfig
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.storage.parquet_store import ParquetStore
from ck_trading.strategies.buy_the_dip import BuyTheDipStrategy

DEFAULT_SAMPLE_PORTFOLIO = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AVGO", "JPM", "V",
]


def resolve_universe(cli_tickers: str | None) -> list[str]:
    if cli_tickers:
        return [t.strip().upper() for t in cli_tickers.split(",") if t.strip()]

    try:
        with MetadataStore() as meta:
            positions = meta.get_open_positions()
        tickers = sorted({p["ticker"] for p in positions if not p["ticker"].endswith(".HK")})
        if tickers:
            print(f"Loaded {len(tickers)} US tickers from your portfolio.")
            return tickers
    except Exception as e:
        print(f"Could not read portfolio from metadata store: {e}")

    print("No portfolio found; using a default sample portfolio.")
    return list(DEFAULT_SAMPLE_PORTFOLIO)


def load_prices(
    tickers: list[str], benchmark: str, start: date, end: date
) -> pl.DataFrame:
    """Load prices for tickers (and benchmark) from Parquet, downloading
    anything missing via yfinance."""
    needed = sorted(set(tickers) | {benchmark})
    store = ParquetStore()
    existing = store.load_prices("us", tickers=needed)

    have: set[str] = set()
    if not existing.is_empty():
        have = set(existing["ticker"].unique().to_list())

    missing = [t for t in needed if t not in have]
    if missing:
        print(f"Downloading prices for {len(missing)} tickers via yfinance: {missing}")
        collector = USMarketCollector()
        fresh = collector.collect_prices(missing, start, end)
        if not fresh.is_empty():
            store.save_prices(fresh, "us")
            existing = pl.concat([existing, fresh], how="diagonal") if not existing.is_empty() else fresh

    if existing.is_empty():
        return existing

    return existing.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated tickers; overrides portfolio")
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=365 * 5),
        help="Backtest start date (default: 5 years ago)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
        help="Backtest end date (default: today)",
    )
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--min-dip-pct", type=float, default=0.10)
    parser.add_argument("--trend-period", type=int, default=200)
    parser.add_argument("--reversal-window", type=int, default=5)
    parser.add_argument("--no-uptrend-filter", action="store_true")
    parser.add_argument(
        "--rebalance",
        choices=("monthly", "quarterly", "annual"),
        default="monthly",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    universe = resolve_universe(args.tickers)
    print(f"Universe ({len(universe)}): {', '.join(universe)}")
    print(f"Period: {args.start} → {args.end}, benchmark={args.benchmark}")

    prices = load_prices(universe, args.benchmark, args.start, args.end)
    if prices.is_empty():
        print("No price data available. Aborting.")
        return 1

    have = set(prices["ticker"].unique().to_list())
    tradable = [t for t in universe if t in have]
    missing = [t for t in universe if t not in have]
    if missing:
        print(f"Skipping (no price data): {missing}")

    strategy = BuyTheDipStrategy(
        lookback=args.lookback,
        min_dip_pct=args.min_dip_pct,
        trend_period=args.trend_period,
        reversal_window=args.reversal_window,
        require_uptrend=not args.no_uptrend_filter,
        top_n=args.max_positions,
    )
    print(f"\nStrategy: {strategy.name}")
    print(f"  {strategy.description}")

    config = BacktestConfig(
        strategy_name=strategy.name,
        universe=tradable,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        rebalance_freq=args.rebalance,
        transaction_cost_bps=args.transaction_cost_bps,
        benchmark=args.benchmark,
    )

    engine = BacktestEngine(strategy, config, prices, fundamentals=pl.DataFrame())
    result = engine.run()

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    print(f"\nTrades: {result.trades.height}")
    if not result.trades.is_empty():
        print(result.trades.tail(10))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
