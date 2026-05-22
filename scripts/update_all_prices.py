"""Update prices for every ticker that has an existing parquet file.

Sweeps data/prices/us/daily/ and refreshes each ticker via yfinance.
Batches the calls (50 tickers per request) and writes incremental progress
to stdout so it's safe to run in the background.
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.storage.parquet_store import ParquetStore

REPO = Path(__file__).resolve().parent.parent
PRICES_DIR = REPO / "data" / "prices" / "us" / "daily"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    tickers = sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))
    if not tickers:
        log("No tickers found in parquet store.")
        return

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=14)

    log(f"Updating {len(tickers)} US tickers, window {start} → {end}")
    collector = USMarketCollector()
    store = ParquetStore()

    batch_size = 50
    saved = 0
    failed: list[str] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        try:
            df = collector.collect_prices(batch, start, end)
            if not df.is_empty():
                store.save_prices(df, "us")
                saved += df["ticker"].n_unique()
        except Exception as e:
            failed.extend(batch)
            log(f"  batch {batch_num}/{total_batches} error: {type(e).__name__}: {e}")

        if batch_num % 5 == 0:
            log(f"  progress {batch_num}/{total_batches} batches  ({saved} tickers saved)")
        time.sleep(0.5)

    log(f"Done. {saved}/{len(tickers)} tickers updated. {len(failed)} failed.")
    if failed[:20]:
        log(f"First failed: {failed[:20]}")


if __name__ == "__main__":
    main()
