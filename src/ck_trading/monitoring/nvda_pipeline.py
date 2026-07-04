"""Weekly orchestration for the NVDA investment framework monitor.

Sections (each isolated, mirroring the att pipeline):

    prices        DAILY closes for the watch tickers via yfinance
                  (daily, not weekly — the technical layer needs
                  60-trading-day highs and the 200-day SMA)
    fundamentals  read-only sanity pass over fundamentals.json
                  (dashboard/backfill are the only writers)
    alerts        evaluate NVDA_RULES -> alerts.json episodes

Nothing here may depend on the gitignored data/prices/ tree (absent in CI
checkouts) — historical seeding lives in scripts/nvda_monitor_update.py
--backfill only.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from ck_trading.config import settings
from ck_trading.monitoring import nvda_metrics
from ck_trading.monitoring.nvda_config import (
    DEFAULT_CHECKLIST,
    NVDA_DEDUPE_KEYS,
    NVDA_RULES,
    WATCH_TICKERS,
)
from ck_trading.monitoring.pipeline import ComponentOutcome, PipelineResult
from ck_trading.monitoring.quarters import (
    EMPTY_SERIES_SCHEMA,
    fundamentals_series,
)
from ck_trading.monitoring.rules import evaluate_rules
from ck_trading.monitoring.store import MonitoringStore

logger = logging.getLogger(__name__)

PRICE_LOOKBACK_DAYS = 30  # daily window re-fetched each run (heals ~4 weeks)


def nvda_store() -> MonitoringStore:
    return MonitoringStore(
        settings.monitoring_dir / "nvda",
        dedupe_keys=NVDA_DEDUPE_KEYS,
        checklist_seed=DEFAULT_CHECKLIST,
    )


def build_nvda_series_provider(store: MonitoringStore):
    """SeriesProvider over the NVDA monitor's stored data.

    Caches the prices frame, the quarters list, and per-ticker indicator
    tables so the three price rules don't recompute rolling windows.
    """
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

    def _indicators(ticker: str) -> pl.DataFrame:
        key = f"ind:{ticker}"
        if key not in cache:
            cache[key] = nvda_metrics.daily_indicators(_prices(), ticker)
        return cache[key]  # type: ignore[return-value]

    def provider(metric_key: str, dims: dict[str, str]) -> pl.DataFrame:
        ticker = dims.get("ticker", "")
        if metric_key == "nvda.gated_dip":
            return nvda_metrics.weekly_sample(_indicators(ticker), "gated_dip")
        if metric_key == "nvda.pct_vs_200dma":
            return nvda_metrics.weekly_sample(
                _indicators(ticker), "pct_vs_200dma"
            )
        if metric_key == "nvda.forward_pe":
            return nvda_metrics.forward_pe_series(
                _prices(), ticker, _quarters()
            )
        if metric_key == "nvda.fundamental":
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
    store = store or nvda_store()
    result = PipelineResult(
        as_of=as_of,
        title="NVDA framework monitor",
        collection_sections=("prices",),
    )

    tickers = [t for t, _role in WATCH_TICKERS]

    # ---- prices (DAILY rows) ------------------------------------------------
    if skip_prices:
        result.outcomes.append(ComponentOutcome("prices", "skipped"))
    else:
        try:
            daily = USMarketCollector().collect_prices(
                tickers,
                as_of - timedelta(days=PRICE_LOOKBACK_DAYS),
                as_of + timedelta(days=1),  # yfinance end is exclusive
            )
            clean = nvda_metrics.normalize_daily(daily)
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
            errors = nvda_metrics.validate_fundamentals(quarters)
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
        provider = build_nvda_series_provider(store)
        rule_results = evaluate_rules(NVDA_RULES, provider)
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
