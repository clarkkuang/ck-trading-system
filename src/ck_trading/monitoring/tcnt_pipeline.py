"""Weekly orchestration for the Tencent (0700.HK) exit monitor — 6th instance.

Sections (isolated, mirroring the other five):

    prices        DAILY closes for 0700.HK and HK peers via yfinance
    fundamentals  read-only sanity pass over fundamentals.json
                  (dashboard / backfill are the only writers)
    alerts        evaluate TCNT_RULES -> alerts.json episodes

The valuation layer is a TRAILING non-IFRS P/E in HKD (tcnt.forward_pe), not
a forward consensus P/E — Tencent reports in RMB and no free consensus feed
exists, so the multiple is built from entered quarterly profit. Nothing here
may depend on the gitignored data/prices/ tree.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from ck_trading.config import settings
from ck_trading.monitoring import tcnt_metrics
from ck_trading.monitoring.pipeline import ComponentOutcome, PipelineResult
from ck_trading.monitoring.quarters import (
    EMPTY_SERIES_SCHEMA,
    fundamentals_series,
)
from ck_trading.monitoring.rules import evaluate_rules
from ck_trading.monitoring.store import MonitoringStore
from ck_trading.monitoring.tcnt_config import (
    DEFAULT_CHECKLIST,
    RELATIVE_BENCHMARK,
    RELATIVE_WINDOW_WEEKS,
    TCNT_DEDUPE_KEYS,
    TCNT_RULES,
    WATCH_TICKERS,
)

logger = logging.getLogger(__name__)

PRICE_LOOKBACK_DAYS = 30
SUBJECT_TICKER = "0700.HK"


def tcnt_store() -> MonitoringStore:
    return MonitoringStore(
        settings.monitoring_dir / "tcnt",
        dedupe_keys=TCNT_DEDUPE_KEYS,
        checklist_seed=DEFAULT_CHECKLIST,
    )


def build_tcnt_series_provider(store: MonitoringStore):
    """SeriesProvider over the Tencent monitor's stored data."""
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
        ticker = dims.get("ticker", "")
        if metric_key == "tcnt.price_weekly_close":
            return tcnt_metrics.weekly_close_series(_prices(), ticker)
        if metric_key == "tcnt.forward_pe":
            return tcnt_metrics.trailing_pe_series(
                _prices(), ticker or SUBJECT_TICKER, _quarters()
            )
        if metric_key == "tcnt.relative_strength":
            return tcnt_metrics.relative_strength_series(
                _prices(),
                ticker or SUBJECT_TICKER,
                dims.get("benchmark", RELATIVE_BENCHMARK),
                RELATIVE_WINDOW_WEEKS,
            )
        if metric_key == "tcnt.fundamental":
            return fundamentals_series(_quarters(), dims.get("field", ""))
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)

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
    store = store or tcnt_store()
    result = PipelineResult(
        as_of=as_of,
        title="Tencent exit monitor",
        collection_sections=("prices",),
    )

    tickers = [t for t, _role in WATCH_TICKERS]

    # ---- prices (DAILY rows) -----------------------------------------------
    if skip_prices:
        result.outcomes.append(ComponentOutcome("prices", "skipped"))
    else:
        try:
            daily = USMarketCollector().collect_prices(
                tickers,
                as_of - timedelta(days=PRICE_LOOKBACK_DAYS),
                as_of + timedelta(days=1),
            )
            clean = tcnt_metrics.normalize_daily(daily)
            if clean.is_empty():
                result.outcomes.append(
                    ComponentOutcome("prices", "failed", error="no price rows")
                )
            else:
                if not dry_run:
                    store.save("prices", clean)
                got = set(clean["ticker"].unique().to_list())
                missing = sorted(set(tickers) - got)
                result.outcomes.append(ComponentOutcome(
                    "prices", "ok", rows=clean.height,
                    extra={"missing": missing} if missing else {},
                ))
        except Exception as e:  # noqa: BLE001
            logger.warning("prices section failed: %r", e)
            result.outcomes.append(
                ComponentOutcome("prices", "failed", error=repr(e))
            )

    # ---- fundamentals (read-only sanity) ------------------------------------
    try:
        doc = store.load_json("fundamentals", default=None)
        quarters = (doc or {}).get("quarters", [])
        if not quarters:
            result.outcomes.append(ComponentOutcome(
                "fundamentals", "empty",
                extra={"reason": "no quarterly data entered yet"},
            ))
        else:
            errors = tcnt_metrics.validate_fundamentals(quarters)
            if errors:
                result.outcomes.append(ComponentOutcome(
                    "fundamentals", "failed", error="; ".join(errors[:3]),
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

    # ---- alerts --------------------------------------------------------------
    try:
        provider = build_tcnt_series_provider(store)
        rule_results = evaluate_rules(TCNT_RULES, provider)
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
