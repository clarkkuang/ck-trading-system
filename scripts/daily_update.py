#!/usr/bin/env python3
"""Daily data update script.

Run daily after US market close (e.g. 5pm ET / 5am CST+1):
  - Update prices for all US tickers in universe
  - Update fundamentals weekly (on Sundays)
  - Log progress and errors
"""

import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.collectors.fundamentals import FundamentalsCollector
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.storage.parquet_store import ParquetStore


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def update_prices() -> int:
    """Update prices for all US universe tickers. Returns count of tickers updated."""
    meta = MetadataStore()
    store = ParquetStore()

    universe = meta.get_universe("US")
    tickers = [t["ticker"] for t in universe]
    meta.close()

    if not tickers:
        log("No tickers in universe. Skipping price update.")
        return 0

    log(f"Updating prices for {len(tickers)} US tickers...")

    collector = USMarketCollector()
    end = date.today()
    start = end - timedelta(days=7)  # 7-day overlap for safety

    saved = 0
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            df = collector.collect_prices(batch, start, end)
            if not df.is_empty():
                store.save_prices(df, "us")
                saved += df["ticker"].n_unique()
        except Exception as e:
            log(f"  Price batch {i // batch_size + 1} error: {e}")
        time.sleep(1)

    log(f"Prices updated: {saved} tickers")
    return saved


def update_fundamentals() -> int:
    """Update fundamentals for all US universe tickers. Returns count updated."""
    meta = MetadataStore()
    store = ParquetStore()

    universe = meta.get_universe("US")
    etfs = {"SPY", "QQQ", "IWM", "VTV", "TLT", "GLD", "DIA", "VOO"}
    tickers = [t["ticker"] for t in universe if t["ticker"] not in etfs]
    meta.close()

    if not tickers:
        log("No tickers for fundamentals update.")
        return 0

    log(f"Updating fundamentals for {len(tickers)} US tickers...")

    collector = FundamentalsCollector()
    saved = 0
    batch_size = 30
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            df = collector.collect(batch)
            if not df.is_empty():
                store.save_fundamentals(df, "us")
                saved += df["ticker"].n_unique()
        except Exception as e:
            log(f"  Fundamentals batch {i // batch_size + 1} error: {e}")
        time.sleep(2)

    log(f"Fundamentals updated: {saved} tickers")
    return saved


def main():
    log("=== Daily Data Update ===")
    t0 = time.time()

    # Always update prices
    price_count = update_prices()

    # Update fundamentals on weekends (Saturday/Sunday) or if --fundamentals flag
    is_weekend = date.today().weekday() >= 5
    force_fund = "--fundamentals" in sys.argv
    if is_weekend or force_fund:
        fund_count = update_fundamentals()
    else:
        log("Skipping fundamentals (weekday). Use --fundamentals to force.")
        fund_count = 0

    elapsed = time.time() - t0
    log(f"=== Done in {elapsed / 60:.1f} min. Prices: {price_count}, Fundamentals: {fund_count} ===")


if __name__ == "__main__":
    main()
