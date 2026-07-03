#!/usr/bin/env python3
"""Weekly AI model share monitor update.

Collects OpenRouter pricing + daily token rankings and npm/PyPI download
stats, derives weekly dollar-weighted bloc shares, evaluates threshold
rules, and persists everything to the git-tracked data/monitoring/ dir.

Run weekly (GitHub Action: .github/workflows/ai-share-monitor.yml) or
manually:

    .venv/bin/python scripts/ai_share_update.py
    .venv/bin/python scripts/ai_share_update.py --skip-rankings --dry-run

Exit codes: 0 = success or partial success (some sections failed but
history still accumulated); 1 = total collection failure.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ck_trading.monitoring.pipeline import run_weekly_update
from ck_trading.storage.metadata_store import MetadataStore


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip the OpenRouter pricing snapshot")
    parser.add_argument("--skip-rankings", action="store_true",
                        help="Skip the OpenRouter rankings dataset")
    parser.add_argument("--skip-downloads", action="store_true",
                        help="Skip npm/PyPI download collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect + evaluate but write nothing")
    args = parser.parse_args()

    log("=== AI Model Share Update ===")

    meta = MetadataStore()
    job_id = meta.log_job_start("ai_share_update")

    try:
        result = run_weekly_update(
            skip_prices=args.skip_prices,
            skip_rankings=args.skip_rankings,
            skip_downloads=args.skip_downloads,
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
