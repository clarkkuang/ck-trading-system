#!/usr/bin/env python
"""CLI for the Tencent (0700.HK) exit monitor.

    python scripts/tcnt_monitor_update.py            # weekly update
    python scripts/tcnt_monitor_update.py --dry-run  # evaluate, write nothing
    python scripts/tcnt_monitor_update.py --backfill # seed prices + fundamentals
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--backfill", action="store_true",
                    help="seed multi-year prices and prefill fundamentals")
    args = ap.parse_args()

    from ck_trading.monitoring.tcnt_config import DEFAULT_FUNDAMENTALS, WATCH_TICKERS
    from ck_trading.monitoring.tcnt_pipeline import run_weekly_update, tcnt_store

    store = tcnt_store()

    if args.backfill:
        from ck_trading.collectors.us_market import USMarketCollector
        from ck_trading.monitoring import tcnt_metrics

        start = date.today() - timedelta(days=365 * 3)
        daily = USMarketCollector().collect_prices(
            [t for t, _ in WATCH_TICKERS], start, date.today() + timedelta(days=1)
        )
        clean = tcnt_metrics.normalize_daily(daily)
        store.save("prices", clean)
        print(f"backfilled {clean.height} price rows from {start}")

        if not (store.load_json("fundamentals") or {}).get("quarters"):
            store.save_json("fundamentals", {
                "version": 1,
                "updated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "quarters": [dict(q) for q in DEFAULT_FUNDAMENTALS],
            })
            print("seeded fundamentals")

    res = run_weekly_update(dry_run=args.dry_run, skip_prices=args.skip_prices)
    print(res.render_text() if hasattr(res, "render_text") else res.title)
    for o in res.outcomes:
        print(f"  {o.name:14s} {o.status:9s} rows={o.rows} {o.error or ''}")
    fired = [r.rule_id for r in res.rule_results if r.fired]
    print(f"  rules: {len(res.rule_results)} evaluated, fired: {fired or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
