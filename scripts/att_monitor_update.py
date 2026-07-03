#!/usr/bin/env python3
"""Weekly AT&T vs SpaceX monitor update.

Collects weekly closes for the watch tickers (T/VZ/TMUS/ASTS/SATS/SPCX),
sanity-checks the manually-entered quarterly fundamentals, evaluates the
threshold rules, and persists to the git-tracked data/monitoring/att/ dir.

Run weekly (GitHub Action: .github/workflows/att-monitor.yml) or manually:

    .venv/bin/python scripts/att_monitor_update.py
    .venv/bin/python scripts/att_monitor_update.py --skip-prices --dry-run

Exit codes: 0 = success or partial; 1 = total collection failure.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ck_trading.monitoring.att_pipeline import run_weekly_update
from ck_trading.storage.metadata_store import MetadataStore


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip the yfinance price fetch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect + evaluate but write nothing")
    args = parser.parse_args()

    log("=== AT&T/SpaceX Monitor Update ===")

    meta = MetadataStore()
    job_id = meta.log_job_start("att_monitor_update")

    try:
        result = run_weekly_update(
            skip_prices=args.skip_prices,
            dry_run=args.dry_run,
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
