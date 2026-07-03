"""Storage for the AI model share monitor.

Everything lives under the git-TRACKED ``data/monitoring/`` directory so the
weekly GitHub Action can commit it and the local dashboard sees the same
state after ``git pull``:

    openrouter_daily_tokens.parquet      raw daily rankings (merge+dedupe)
    openrouter_pricing_snapshots.parquet raw pricing snapshots (merge+dedupe)
    pkg_downloads.parquet                npm/pypi weekly downloads (merge+dedupe)
    weekly_bloc_share.parquet            derived; overwritten every run
    alerts.json                          rule episode state (canonical)
    checklist.json                       L2/L3 manual checklist (dashboard-only;
                                         the pipeline/CI NEVER writes this file)

JSON files serialize deterministically (indent=2, sorted keys, trailing
newline) so CI commits produce clean diffs.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from ck_trading.config import settings
from ck_trading.monitoring.rules import EvalResult

DEDUPE_KEYS: dict[str, list[str]] = {
    "openrouter_daily_tokens": ["date", "model_id", "scope"],
    "openrouter_pricing_snapshots": ["snapshot_date", "model_id"],
    "pkg_downloads": ["registry", "package", "iso_week"],
}

_HISTORY_CAP = 52  # weeks of per-rule history kept in alerts.json


class MonitoringStore:
    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        dedupe_keys: dict[str, list[str]] | None = None,
        checklist_seed: tuple[dict, ...] | list[dict] | None = None,
    ):
        """A monitor's storage root.

        Args:
            base_dir: storage directory (default: settings.monitoring_dir).
                Sub-monitors pass a subdirectory, e.g. monitoring_dir/"att".
            dedupe_keys: per-table dedupe keys; None falls back to the
                module-level DEDUPE_KEYS (the AI share monitor tables).
            checklist_seed: default checklist items seeded by
                load_checklist() when no file exists; None falls back to
                rules_config.DEFAULT_CHECKLIST.
        """
        self.base_dir = Path(base_dir) if base_dir else settings.monitoring_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._dedupe_keys = dedupe_keys if dedupe_keys is not None else DEDUPE_KEYS
        self._checklist_seed = checklist_seed

    # ------------------------------------------------------------------
    # Parquet
    # ------------------------------------------------------------------
    def _path(self, name: str) -> Path:
        return self.base_dir / f"{name}.parquet"

    def save(self, name: str, df: pl.DataFrame) -> None:
        """Merge-dedupe into {name}.parquet (keep latest row per key)."""
        if df.is_empty():
            return
        keys = self._dedupe_keys.get(name)
        path = self._path(name)
        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df], how="diagonal")
        if keys:
            df = df.unique(subset=keys, keep="last")
        sort_col = keys[0] if keys else df.columns[0]
        df.sort(sort_col).write_parquet(path)

    def overwrite(self, name: str, df: pl.DataFrame) -> None:
        """Replace {name}.parquet entirely (for derived tables)."""
        df.write_parquet(self._path(name))

    def load(self, name: str) -> pl.DataFrame:
        path = self._path(name)
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    # ------------------------------------------------------------------
    # Alerts (episode state)
    # ------------------------------------------------------------------
    @property
    def alerts_path(self) -> Path:
        return self.base_dir / "alerts.json"

    def load_alerts(self) -> dict:
        if not self.alerts_path.exists():
            return {}
        try:
            return json.loads(self.alerts_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def save_alerts(self, results: list[EvalResult]) -> dict:
        """Merge evaluation results into episode state and persist.

        Episode semantics:
            fired & no active episode  -> new episode (first_period_key set)
            fired & active episode     -> update last_period_key/value only
            not fired & active episode -> mark resolved
        Same-period re-runs are idempotent.

        Returns the persisted state dict.
        """
        prev = self.load_alerts()
        prev_rules = {r["rule_id"]: r for r in prev.get("rules", [])}
        now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        out_rules: list[dict] = []
        for res in results:
            entry = asdict(res)
            entry["fired"] = res.fired
            old = prev_rules.get(res.rule_id, {})

            # ---- episode bookkeeping
            if res.fired:
                if old.get("status") == "triggered":
                    # ongoing episode: keep the original start
                    entry["episode_started"] = old.get(
                        "episode_started", res.first_period_key
                    )
                    entry["triggered_at"] = old.get("triggered_at", now_iso)
                else:
                    entry["episode_started"] = res.first_period_key
                    entry["triggered_at"] = now_iso
                entry["resolved_at"] = None
            else:
                entry["episode_started"] = None
                entry["triggered_at"] = None
                if old.get("status") == "triggered":
                    entry["resolved_at"] = now_iso
                else:
                    entry["resolved_at"] = old.get("resolved_at")

            # ---- rolling per-rule history
            history = list(old.get("history", []))
            point = {
                "period_key": res.period_key,
                "value": res.metric_value,
                "status": res.status,
            }
            if history and history[-1].get("period_key") == res.period_key:
                history[-1] = point  # same-period re-run: replace
            else:
                history.append(point)
            entry["history"] = history[-_HISTORY_CAP:]

            out_rules.append(entry)

        state = {"updated_at": now_iso, "rules": out_rules}
        self.alerts_path.write_text(
            json.dumps(state, indent=2, sort_keys=True, default=str) + "\n"
        )
        return state

    # ------------------------------------------------------------------
    # Manual checklist (dashboard-only writer)
    # ------------------------------------------------------------------
    @property
    def checklist_path(self) -> Path:
        return self.base_dir / "checklist.json"

    def load_checklist(self) -> list[dict]:
        """Load checklist items, seeding defaults on first use (never writes)."""
        if self.checklist_path.exists():
            try:
                return json.loads(self.checklist_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        seed = self._checklist_seed
        if seed is None:
            from ck_trading.monitoring.rules_config import DEFAULT_CHECKLIST

            seed = DEFAULT_CHECKLIST
        return [{**item, "last_checked": None, "notes": ""} for item in seed]

    def save_checklist(self, items: list[dict]) -> None:
        self.checklist_path.write_text(
            json.dumps(items, indent=2, sort_keys=True, default=str) + "\n"
        )

    # ------------------------------------------------------------------
    # Generic JSON documents (e.g. fundamentals.json, scenarios.json)
    # ------------------------------------------------------------------
    def load_json(self, name: str, default=None):
        """Read {name}.json; `default` on missing/corrupt (same tolerance
        as load_alerts)."""
        path = self.base_dir / f"{name}.json"
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return default

    def save_json(self, name: str, obj) -> None:
        """Deterministic write: indent=2, sorted keys, trailing newline."""
        path = self.base_dir / f"{name}.json"
        path.write_text(
            json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
        )
