"""Tests for the AT&T monitor pipeline."""

import datetime as dt
from datetime import date
from unittest.mock import patch

import polars as pl
import pytest

from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.monitoring.att_config import ATT_DEDUPE_KEYS, DEFAULT_CHECKLIST
from ck_trading.monitoring.att_pipeline import (
    build_att_series_provider,
    run_weekly_update,
)
from ck_trading.monitoring.store import MonitoringStore

AS_OF = date(2026, 7, 3)


def _att_store(tmp_path) -> MonitoringStore:
    return MonitoringStore(
        tmp_path, dedupe_keys=ATT_DEDUPE_KEYS, checklist_seed=DEFAULT_CHECKLIST
    )


def _daily_prices() -> pl.DataFrame:
    """Two weeks of daily closes for all six tickers; T well above $16."""
    rows = []
    closes = {"T": 20.7, "VZ": 40.0, "TMUS": 230.0, "ASTS": 45.0,
              "SATS": 28.0, "SPCX": 162.0}
    for tk, base in closes.items():
        for d in (date(2026, 6, 22), date(2026, 6, 24),
                  date(2026, 6, 29), date(2026, 7, 2)):
            rows.append({"ticker": tk, "date": d, "close": base})
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "date": pl.Date, "close": pl.Float64}
    )


def _outcome(result, name):
    return next(o for o in result.outcomes if o.name == name)


@pytest.fixture
def patched_prices():
    with patch.object(
        USMarketCollector, "collect_prices", return_value=_daily_prices()
    ) as m:
        yield m


class TestAttPipeline:
    def test_happy_path(self, tmp_path, patched_prices):
        store = _att_store(tmp_path)
        result = run_weekly_update(as_of=AS_OF, store=store)

        assert _outcome(result, "prices").status == "ok"
        assert _outcome(result, "fundamentals").status == "empty"
        assert _outcome(result, "alerts").status == "ok"
        assert not result.total_failure
        assert "AT&T/SpaceX monitor" in result.summary()

        prices = store.load("prices")
        assert not prices.is_empty()
        assert set(prices["ticker"].unique().to_list()) == {
            "T", "VZ", "TMUS", "ASTS", "SATS", "SPCX"
        }
        assert store.alerts_path.exists()

    def test_first_run_rule_statuses(self, tmp_path, patched_prices):
        """Empty fundamentals -> quarterly rules insufficient; price rules live."""
        store = _att_store(tmp_path)
        result = run_weekly_update(as_of=AS_OF, store=store)
        by_id = {r.rule_id: r for r in result.rule_results}

        # price rules evaluated (T at 20.7 -> ok, not triggered)
        assert by_id["att_price_below_16_structural"].status == "ok"
        assert by_id["att_price_below_12_tail"].status == "ok"
        # fundamentals rules lack data
        for rid in ("att_churn_ge_095", "att_churn_ge_110",
                    "att_churn_rising_3q", "att_convergence_stalled_2q",
                    "att_fcf_below_4b"):
            assert by_id[rid].status == "insufficient_data", rid

    def test_price_below_16_triggers(self, tmp_path):
        cheap = _daily_prices().with_columns(
            pl.when(pl.col("ticker") == "T")
            .then(15.2).otherwise(pl.col("close")).alias("close")
        )
        with patch.object(USMarketCollector, "collect_prices", return_value=cheap):
            store = _att_store(tmp_path)
            result = run_weekly_update(as_of=AS_OF, store=store)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["att_price_below_16_structural"].status == "triggered"
        assert by_id["att_price_below_12_tail"].status == "ok"

    def test_fundamentals_rules_fire_when_entered(self, tmp_path, patched_prices):
        store = _att_store(tmp_path)
        store.save_json("fundamentals", {"version": 1, "quarters": [
            {"quarter": "2025Q3", "churn_pct": 0.85, "convergence_pct": 45.0,
             "fcf_billions": 4.5},
            {"quarter": "2025Q4", "churn_pct": 0.95, "convergence_pct": 45.0,
             "fcf_billions": 4.2},
            {"quarter": "2026Q1", "churn_pct": 1.12, "convergence_pct": 44.0,
             "fcf_billions": 3.8},
        ]})
        result = run_weekly_update(as_of=AS_OF, store=store)
        by_id = {r.rule_id: r for r in result.rule_results}

        assert _outcome(result, "fundamentals").status == "ok"
        assert by_id["att_churn_ge_095"].status == "triggered"   # 1.12 >= 0.95
        assert by_id["att_churn_ge_110"].status == "triggered"   # 1.12 >= 1.10
        # churn rising: 0.85 -> 0.95 -> 1.12 = only 2 comparisons, need 3
        assert by_id["att_churn_rising_3q"].status == "insufficient_data"
        # convergence 45 -> 45 (flat counts) -> 44 (down) = 2 non-increasing
        assert by_id["att_convergence_stalled_2q"].status == "triggered"
        assert by_id["att_fcf_below_4b"].status == "triggered"   # 3.8 < 4.0

    def test_malformed_fundamentals_flagged(self, tmp_path, patched_prices):
        store = _att_store(tmp_path)
        store.save_json("fundamentals", {"quarters": [{"quarter": "26Q1"}]})
        result = run_weekly_update(as_of=AS_OF, store=store)
        assert _outcome(result, "fundamentals").status == "failed"

    def test_prices_failure_isolated(self, tmp_path):
        with patch.object(
            USMarketCollector, "collect_prices",
            side_effect=RuntimeError("yf down"),
        ):
            store = _att_store(tmp_path)
            # pre-seed a prior week so alerts can still evaluate history
            store.save("prices", pl.DataFrame({
                "ticker": ["T"], "iso_week": ["2026-W26"],
                "week_start": [date(2026, 6, 22)],
                "price_date": [date(2026, 6, 26)], "close": [20.5],
                "collected_at": [dt.datetime(2026, 6, 26)],
            }))
            result = run_weekly_update(as_of=AS_OF, store=store)
        assert _outcome(result, "prices").status == "failed"
        assert result.total_failure  # prices is the only collection section
        assert _outcome(result, "alerts").status == "ok"  # still evaluated
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["att_price_below_16_structural"].status == "ok"

    def test_dry_run_writes_nothing(self, tmp_path, patched_prices):
        store = _att_store(tmp_path)
        run_weekly_update(as_of=AS_OF, store=store, dry_run=True)
        assert store.load("prices").is_empty()
        assert not store.alerts_path.exists()

    def test_rerun_idempotent(self, tmp_path, patched_prices):
        store = _att_store(tmp_path)
        run_weekly_update(as_of=AS_OF, store=store)
        n1 = store.load("prices").height
        run_weekly_update(as_of=AS_OF, store=store)
        assert store.load("prices").height == n1

    def test_missing_ticker_tolerated(self, tmp_path):
        partial = _daily_prices().filter(pl.col("ticker") != "SPCX")
        with patch.object(
            USMarketCollector, "collect_prices", return_value=partial
        ):
            store = _att_store(tmp_path)
            result = run_weekly_update(as_of=AS_OF, store=store)
        o = _outcome(result, "prices")
        assert o.status == "ok"
        assert o.extra["missing"] == ["SPCX"]


class TestAttSeriesProvider:
    def test_price_series_routing(self, tmp_path):
        store = _att_store(tmp_path)
        store.save("prices", pl.DataFrame({
            "ticker": ["T"], "iso_week": ["2026-W27"],
            "week_start": [date(2026, 6, 29)],
            "price_date": [date(2026, 7, 2)], "close": [20.7],
            "collected_at": [dt.datetime(2026, 7, 3)],
        }))
        provider = build_att_series_provider(store)
        s = provider("att.price_weekly_close", {"ticker": "T"})
        assert s["value"].to_list() == [20.7]

    def test_fundamental_routing(self, tmp_path):
        store = _att_store(tmp_path)
        store.save_json("fundamentals", {"quarters": [
            {"quarter": "2026Q1", "churn_pct": 0.83},
        ]})
        provider = build_att_series_provider(store)
        s = provider("att.fundamental", {"field": "churn_pct"})
        assert s["value"].to_list() == [0.83]

    def test_unknown_metric_empty(self, tmp_path):
        provider = build_att_series_provider(_att_store(tmp_path))
        assert provider("nope", {}).is_empty()
