"""Weekly orchestration for the AT&T vs SpaceX telecom threat monitor.

``run_weekly_update()`` is shared by the CLI (scripts/att_monitor_update.py,
also the GitHub Action entry) and the dashboard's "Run collection now"
button. Sections run independently:

    prices        weekly closes for the watch tickers via yfinance
    fundamentals  read-only sanity pass over fundamentals.json
                  (the dashboard is the ONLY writer)
    alerts        evaluate ATT_RULES -> alerts.json episodes

The monitor stores everything under the git-tracked data/monitoring/att/
directory — data/prices/ is gitignored and absent in CI checkouts, so
nothing here may depend on it.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from ck_trading.config import settings
from ck_trading.monitoring import att_metrics
from ck_trading.monitoring.att_config import (
    ATT_DEDUPE_KEYS,
    ATT_RULES,
    DEFAULT_CHECKLIST,
    WATCH_TICKERS,
)
from ck_trading.monitoring.pipeline import ComponentOutcome, PipelineResult
from ck_trading.monitoring.rules import evaluate_rules
from ck_trading.monitoring.store import MonitoringStore

logger = logging.getLogger(__name__)

PRICE_LOOKBACK_DAYS = 30  # daily window re-fetched each run (heals ~4 weeks)


def att_store() -> MonitoringStore:
    return MonitoringStore(
        settings.monitoring_dir / "att",
        dedupe_keys=ATT_DEDUPE_KEYS,
        checklist_seed=DEFAULT_CHECKLIST,
    )


def build_att_series_provider(store: MonitoringStore):
    """SeriesProvider over the AT&T monitor's stored data."""
    cache: dict[str, object] = {}

    def _prices() -> pl.DataFrame:
        if "prices" not in cache:
            cache["prices"] = store.load("prices")
        return cache["prices"]  # type: ignore[return-value]

    def _quarters() -> list[dict]:
        if "quarters" not in cache:
            doc = store.load_json("fundamentals", default={}) or {}
            cache["quarters"] = doc.get("quarters", [])
        return cache["quarters"]  # type: ignore[return-value]

    def provider(metric_key: str, dims: dict[str, str]) -> pl.DataFrame:
        if metric_key == "att.price_weekly_close":
            return att_metrics.weekly_close_series(
                _prices(), dims.get("ticker", "")
            )
        if metric_key == "att.fundamental":
            return att_metrics.fundamentals_series(
                _quarters(), dims.get("field", "")
            )
        return pl.DataFrame(schema={
            "period_key": pl.Utf8, "period_start": pl.Date, "value": pl.Float64,
        })

    return provider


def run_weekly_update(
    *,
    skip_prices: bool = False,
    dry_run: bool = False,
    as_of: date | None = None,
    store: MonitoringStore | None = None,
) -> PipelineResult:
    from ck_trading.collectors.us_market import USMarketCollector

    as_of = as_of or date.today()
    store = store or att_store()
    result = PipelineResult(
        as_of=as_of,
        title="AT&T/SpaceX monitor",
        collection_sections=("prices",),
    )

    tickers = [t for t, _role in WATCH_TICKERS]

    # ---- prices ------------------------------------------------------------
    if skip_prices:
        result.outcomes.append(ComponentOutcome("prices", "skipped"))
    else:
        try:
            daily = USMarketCollector().collect_prices(
                tickers,
                as_of - timedelta(days=PRICE_LOOKBACK_DAYS),
                as_of + timedelta(days=1),  # yfinance end is exclusive
            )
            weekly = att_metrics.compute_weekly_closes(daily)
            if weekly.is_empty():
                result.outcomes.append(
                    ComponentOutcome("prices", "failed", error="no price rows")
                )
            else:
                if not dry_run:
                    store.save("prices", weekly)
                got = set(weekly["ticker"].unique().to_list())
                missing = sorted(set(tickers) - got)
                result.outcomes.append(ComponentOutcome(
                    "prices", "ok", rows=weekly.height,
                    extra={"missing": missing} if missing else {},
                ))
        except Exception as e:  # noqa: BLE001
            logger.warning("prices section failed: %r", e)
            result.outcomes.append(
                ComponentOutcome("prices", "failed", error=repr(e))
            )

    # ---- fundamentals (read-only sanity; dashboard is the writer) ----------
    try:
        doc = store.load_json("fundamentals", default=None)
        quarters = (doc or {}).get("quarters", [])
        if not quarters:
            result.outcomes.append(ComponentOutcome(
                "fundamentals", "empty",
                extra={"reason": "no quarterly data entered yet"},
            ))
        else:
            errors = att_metrics.validate_fundamentals(quarters)
            if errors:
                result.outcomes.append(ComponentOutcome(
                    "fundamentals", "failed",
                    error="; ".join(errors[:3]),
                ))
            else:
                result.outcomes.append(ComponentOutcome(
                    "fundamentals", "ok", rows=len(quarters)
                ))
    except Exception as e:  # noqa: BLE001
        logger.warning("fundamentals section failed: %r", e)
        result.outcomes.append(
            ComponentOutcome("fundamentals", "failed", error=repr(e))
        )

    # ---- alerts -------------------------------------------------------------
    try:
        provider = build_att_series_provider(store)
        rule_results = evaluate_rules(ATT_RULES, provider)
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
