#!/usr/bin/env python3
"""Weekly NFLX framework monitor update.

Collects DAILY closes for the watch tickers (NFLX/DIS/SPY/QQQ),
sanity-checks the quarterly fundamentals, evaluates the three-layer
buy/sell rules, and persists to git-tracked data/monitoring/nflx/.

Run weekly (GitHub Action: .github/workflows/nflx-monitor.yml) or manually:

    .venv/bin/python scripts/nflx_monitor_update.py
    .venv/bin/python scripts/nflx_monitor_update.py --backfill   # one-off seeding

--backfill (local-only, one-off):
    * seeds full daily price history: from the local gitignored
      data/prices/us/daily/*.parquet when present, else full-history
      yfinance fetch (all four have local files; fallback kept for robustness)
    * seeds fundamentals.json / scenarios.json from nflx_config defaults
      (only when the files don't exist — never overwrites)
    * then falls through to the normal weekly update
Without seeding, CI would need ~200 trading days to warm up the SMA200.

Exit codes: 0 = success or partial; 1 = total collection failure.
"""

import argparse
import sys
from datetime import date, datetime
from datetime import timezone as dt_timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ck_trading.monitoring.nflx_pipeline import nflx_store, run_weekly_update
from ck_trading.storage.metadata_store import MetadataStore


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def backfill(store) -> None:
    import polars as pl

    from ck_trading.collectors.us_market import USMarketCollector
    from ck_trading.config import settings
    from ck_trading.monitoring.nflx_config import (
        DEFAULT_FUNDAMENTALS,
        DEFAULT_SCENARIOS,
        WATCH_TICKERS,
    )
    from ck_trading.monitoring.nflx_metrics import normalize_daily

    local_dir = settings.prices_dir / "us" / "daily"
    collector = USMarketCollector()

    for ticker, _role in WATCH_TICKERS:
        local = local_dir / f"{ticker}.parquet"
        if local.exists():
            df = pl.read_parquet(local)
            log(f"  backfill {ticker}: {df.height} rows from local parquet")
        else:
            log(f"  backfill {ticker}: no local file — fetching via yfinance")
            df = collector.collect_prices(
                [ticker], date(2016, 1, 1), date.today()
            )
        clean = normalize_daily(df)
        if clean.is_empty():
            log(f"  backfill {ticker}: NO DATA — skipped")
            continue
        store.save("prices", clean)

    if store.load_json("fundamentals") is None:
        store.save_json("fundamentals", {
            "version": 1,
            "updated_at": datetime.now(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quarters": [dict(q) for q in DEFAULT_FUNDAMENTALS],
        })
        log(f"  seeded fundamentals.json ({len(DEFAULT_FUNDAMENTALS)} quarters, thru Q2'26 earnings)")
    if store.load_json("scenarios") is None:
        store.save_json("scenarios", {
            "version": 1,
            "updated_at": datetime.now(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scenarios": [dict(s) for s in DEFAULT_SCENARIOS],
        })
        log("  seeded scenarios.json (weighted fair $81.25)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip the yfinance price fetch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect + evaluate but write nothing")
    parser.add_argument("--backfill", action="store_true",
                        help="One-off local seeding of full price history + jsons")
    args = parser.parse_args()

    log("=== NFLX Framework Monitor Update ===")

    store = nflx_store()
    if args.backfill:
        log("Backfilling full history…")
        backfill(store)

    meta = MetadataStore()
    job_id = meta.log_job_start("nflx_monitor_update")

    try:
        result = run_weekly_update(
            skip_prices=args.skip_prices,
            dry_run=args.dry_run,
            store=store,
        )
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e!r}")
        meta.log_job_end(job_id, "failed", repr(e))
        meta.close()
        return 1

    for line in result.summary().splitlines():
        log(line)

    statuses = {o.name: o.status for o in result.outcomes}
    if result.total_failure:
        job_status = "failed"
    elif all(s in ("ok", "skipped") for s in statuses.values()):
        job_status = "success"
    else:
        job_status = "partial"

    meta.log_job_end(job_id, job_status, result.to_json())
    meta.close()

    log(f"=== Done ({job_status}) ===")
    return 1 if result.total_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
