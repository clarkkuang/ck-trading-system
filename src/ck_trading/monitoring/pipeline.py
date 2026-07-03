"""Weekly orchestration for the AI model share monitor.

``run_weekly_update()`` is the single entry point shared by the CLI script
(scripts/ai_share_update.py, also used by the GitHub Action) and the
dashboard's "Run collection now" button.

Sections run independently — one failing section never kills the others:

    pricing   OpenRouter /models snapshot (no key needed)
    rankings  OpenRouter datasets rankings-daily (skipped without key)
    npm       npm weekly downloads
    pypi      PyPI weekly downloads
    derive    rebuild weekly_bloc_share from FULL stored history
    alerts    evaluate threshold rules -> alerts.json episodes
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import polars as pl

from ck_trading.config import settings
from ck_trading.monitoring import metrics
from ck_trading.monitoring.rules import EvalResult, evaluate_rules
from ck_trading.monitoring.rules_config import PKG_WATCHLIST, RULES
from ck_trading.monitoring.store import MonitoringStore

logger = logging.getLogger(__name__)

RANKINGS_LOOKBACK_DAYS = 10  # tail window re-fetched each run (gap healing)


@dataclass
class ComponentOutcome:
    name: str
    status: str  # "ok" | "skipped" | "empty" | "failed"
    rows: int = 0
    error: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    as_of: date
    outcomes: list[ComponentOutcome] = field(default_factory=list)
    rule_results: list[EvalResult] = field(default_factory=list)
    title: str = "AI share monitor"
    collection_sections: tuple[str, ...] = ("pricing", "rankings", "npm", "pypi")

    @property
    def total_failure(self) -> bool:
        collection = [
            o for o in self.outcomes
            if o.name in self.collection_sections and o.status != "skipped"
        ]
        return bool(collection) and all(o.status == "failed" for o in collection)

    def summary(self) -> str:
        lines = [f"{self.title} — {self.as_of}"]
        for o in self.outcomes:
            line = f"  {o.name:10} {o.status}"
            if o.rows:
                line += f" ({o.rows} rows)"
            if o.error:
                line += f" — {o.error}"
            lines.append(line)
        for r in self.rule_results:
            mark = {"triggered": "🔴", "ok": "🟢"}.get(r.status, "🟡")
            val = f"{r.metric_value:.4g}" if r.metric_value is not None else "—"
            lines.append(
                f"  {mark} {r.rule_id}: {r.status} (value={val}, "
                f"streak {r.streak}/{r.required})"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "as_of": str(self.as_of),
            "outcomes": [
                {"name": o.name, "status": o.status, "rows": o.rows,
                 "error": o.error, **({"extra": o.extra} if o.extra else {})}
                for o in self.outcomes
            ],
            "rules": [
                {"rule_id": r.rule_id, "status": r.status,
                 "value": r.metric_value, "streak": r.streak}
                for r in self.rule_results
            ],
        }, default=str)


def run_weekly_update(
    *,
    skip_prices: bool = False,
    skip_rankings: bool = False,
    skip_downloads: bool = False,
    dry_run: bool = False,
    as_of: date | None = None,
    store: MonitoringStore | None = None,
) -> PipelineResult:
    from ck_trading.collectors.openrouter import OpenRouterCollector
    from ck_trading.collectors.pkg_downloads import PackageDownloadsCollector

    as_of = as_of or date.today()
    store = store or MonitoringStore()
    result = PipelineResult(as_of=as_of)

    or_collector = OpenRouterCollector(api_key=settings.openrouter_api_key)
    pkg_collector = PackageDownloadsCollector()

    # ---- pricing ----------------------------------------------------------
    if skip_prices:
        result.outcomes.append(ComponentOutcome("pricing", "skipped"))
    else:
        try:
            pricing = or_collector.collect_pricing(as_of)
            if pricing.is_empty():
                result.outcomes.append(ComponentOutcome("pricing", "failed",
                                                        error="empty response"))
            else:
                if not dry_run:
                    store.save("openrouter_pricing_snapshots", pricing)
                result.outcomes.append(
                    ComponentOutcome("pricing", "ok", rows=pricing.height)
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("pricing section failed: %r", e)
            result.outcomes.append(
                ComponentOutcome("pricing", "failed", error=repr(e))
            )

    # ---- rankings ---------------------------------------------------------
    if skip_rankings:
        result.outcomes.append(ComponentOutcome("rankings", "skipped"))
    elif not settings.openrouter_api_key:
        result.outcomes.append(ComponentOutcome(
            "rankings", "skipped", extra={"reason": "no OPENROUTER_API_KEY"}
        ))
    else:
        try:
            start = as_of - timedelta(days=RANKINGS_LOOKBACK_DAYS)
            rankings = or_collector.collect_rankings(start, as_of)
            if rankings.is_empty():
                result.outcomes.append(ComponentOutcome("rankings", "empty"))
            else:
                unclassified = (
                    rankings.filter(pl.col("bloc") == "unclassified")["org"]
                    .unique().to_list()
                )
                if not dry_run:
                    store.save("openrouter_daily_tokens", rankings)
                result.outcomes.append(ComponentOutcome(
                    "rankings", "ok", rows=rankings.height,
                    extra={"unclassified_orgs": unclassified} if unclassified else {},
                ))
        except Exception as e:  # noqa: BLE001
            logger.warning("rankings section failed: %r", e)
            result.outcomes.append(
                ComponentOutcome("rankings", "failed", error=repr(e))
            )

    # ---- downloads --------------------------------------------------------
    npm_pkgs = [p for reg, p, _ in PKG_WATCHLIST if reg == "npm"]
    pypi_pkgs = [p for reg, p, _ in PKG_WATCHLIST if reg == "pypi"]

    if skip_downloads:
        result.outcomes.append(ComponentOutcome("npm", "skipped"))
        result.outcomes.append(ComponentOutcome("pypi", "skipped"))
    else:
        for name, fn, pkgs in (
            ("npm", pkg_collector.collect_npm, npm_pkgs),
            ("pypi", pkg_collector.collect_pypi, pypi_pkgs),
        ):
            try:
                df = fn(pkgs, as_of)
                if df.is_empty():
                    result.outcomes.append(ComponentOutcome(name, "failed",
                                                            error="no packages returned"))
                else:
                    if not dry_run:
                        store.save("pkg_downloads", df)
                    missing = sorted(set(pkgs) - set(df["package"].to_list()))
                    result.outcomes.append(ComponentOutcome(
                        name, "ok", rows=df.height,
                        extra={"missing": missing} if missing else {},
                    ))
            except Exception as e:  # noqa: BLE001
                logger.warning("%s section failed: %r", name, e)
                result.outcomes.append(
                    ComponentOutcome(name, "failed", error=repr(e))
                )

    # ---- derive -----------------------------------------------------------
    try:
        daily = store.load("openrouter_daily_tokens")
        pricing_hist = store.load("openrouter_pricing_snapshots")
        weekly = metrics.compute_weekly_bloc_share(daily, pricing_hist)
        if not dry_run and not weekly.is_empty():
            store.overwrite("weekly_bloc_share", weekly)
        result.outcomes.append(
            ComponentOutcome("derive",
                             "ok" if not weekly.is_empty() else "empty",
                             rows=weekly.height)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("derive section failed: %r", e)
        result.outcomes.append(ComponentOutcome("derive", "failed", error=repr(e)))

    # ---- alerts -----------------------------------------------------------
    try:
        provider = build_series_provider(store)
        rule_results = evaluate_rules(RULES, provider)
        result.rule_results = rule_results
        if not dry_run:
            store.save_alerts(rule_results)
        fired = [r.rule_id for r in rule_results if r.fired]
        result.outcomes.append(ComponentOutcome(
            "alerts", "ok", rows=len(rule_results),
            extra={"fired": fired} if fired else {},
        ))
    except Exception as e:  # noqa: BLE001
        logger.warning("alerts section failed: %r", e)
        result.outcomes.append(ComponentOutcome("alerts", "failed", error=repr(e)))

    return result


def build_series_provider(store: MonitoringStore):
    """SeriesProvider over the stored parquet tables (lazily cached)."""
    cache: dict[str, pl.DataFrame] = {}

    def _table(name: str) -> pl.DataFrame:
        if name not in cache:
            cache[name] = store.load(name)
        return cache[name]

    def provider(metric_key: str, dims: dict[str, str]) -> pl.DataFrame:
        empty = pl.DataFrame(schema={
            "period_key": pl.Utf8, "period_start": pl.Date, "value": pl.Float64,
        })
        if metric_key == "openrouter.weekly_dollar_share":
            return metrics.bloc_share_series(
                _table("weekly_bloc_share"),
                bloc=dims.get("bloc", ""), scope=dims.get("scope", ""),
                metric="dollar_share",
            )
        if metric_key == "openrouter.weekly_token_share":
            return metrics.bloc_share_series(
                _table("weekly_bloc_share"),
                bloc=dims.get("bloc", ""), scope=dims.get("scope", ""),
                metric="token_share",
            )
        if metric_key == "openrouter.flagship_completion_price":
            return metrics.flagship_price_series(
                _table("openrouter_pricing_snapshots"),
                family=dims.get("family", ""),
            )
        if metric_key == "pkg.weekly_downloads":
            return metrics.pkg_weekly_series(
                _table("pkg_downloads"),
                registry=dims.get("registry", ""),
                package=dims.get("package", ""),
            )
        return empty

    return provider
