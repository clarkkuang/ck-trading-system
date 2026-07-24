"""Tests for the INTC framework monitor (config, metrics, pipeline)."""

import datetime as dt
from datetime import date, timedelta
from unittest.mock import patch

import polars as pl
import pytest

from ck_trading.monitoring.intc_config import (
    DEFAULT_CHECKLIST,
    DEFAULT_FUNDAMENTALS,
    DEFAULT_SCENARIOS,
    INTC_DEDUPE_KEYS,
    INTC_RULES,
)
from ck_trading.monitoring.intc_metrics import (
    pb_ratio_series,
    validate_fundamentals,
)
from ck_trading.monitoring.intc_pipeline import (
    build_intc_series_provider,
    intc_store,
    run_weekly_update,
)
from ck_trading.monitoring.quarters import weighted_fair_value
from ck_trading.monitoring.store import MonitoringStore


def _store(tmp_path) -> MonitoringStore:
    return MonitoringStore(
        tmp_path, dedupe_keys=INTC_DEDUPE_KEYS, checklist_seed=DEFAULT_CHECKLIST
    )


def _daily_prices(ticker: str, end: date, n: int, closes_fn) -> pl.DataFrame:
    """n trading-ish days ending at `end` with close = closes_fn(i), i=0 oldest."""
    rows = []
    d = end
    i = n - 1
    while i >= 0:
        if d.weekday() < 5:
            rows.append({"ticker": ticker, "date": d, "close": float(closes_fn(i))})
            i -= 1
        d -= timedelta(days=1)
    return pl.DataFrame(rows).sort("date")


def _seed_fundamentals(store: MonitoringStore, quarters=None):
    store.save_json("fundamentals", {
        "version": 1,
        "updated_at": "2026-07-23T00:00:00Z",
        "quarters": list(quarters if quarters is not None else DEFAULT_FUNDAMENTALS),
    })


class TestConfig:
    def test_scenarios_valid_and_weighted_fair(self):
        assert weighted_fair_value(list(DEFAULT_SCENARIOS)) == pytest.approx(83.20)
        probs = sum(s["probability_pct"] for s in DEFAULT_SCENARIOS)
        assert probs == pytest.approx(100.0)

    def test_rules_wellformed_and_prefixed(self):
        ids = [r.rule_id for r in INTC_RULES]
        assert len(ids) == len(set(ids))
        for r in INTC_RULES:
            assert r.rule_id.startswith(("intc_buy_", "intc_sell_"))

    def test_prefill_fundamentals_validate(self):
        assert validate_fundamentals(list(DEFAULT_FUNDAMENTALS)) == []


class TestValidator:
    def test_negative_op_loss_allowed(self):
        # foundry op loss is signed (negative IS the value)
        assert validate_fundamentals(
            [{"quarter": "2026Q3", "foundry_op_loss_billions": -2.0}]
        ) == []

    def test_negative_dcai_revenue_rejected(self):
        assert any(
            "dcai_revenue" in e
            for e in validate_fundamentals(
                [{"quarter": "2026Q3", "dcai_revenue_billions": -1.0}]
            )
        )

    def test_yield_range(self):
        assert any(
            "yield_18a" in e
            for e in validate_fundamentals(
                [{"quarter": "2026Q3", "yield_18a_profitable_pct": 120.0}]
            )
        )

    def test_negative_bvps_rejected(self):
        assert any(
            "book_value" in e
            for e in validate_fundamentals(
                [{"quarter": "2026Q3", "book_value_per_share": -5.0}]
            )
        )

    def test_bad_label(self):
        assert validate_fundamentals([{"quarter": "FY26Q1"}])


class TestPBRatio:
    def test_pb_from_book_value(self):
        # BVPS entered for 2026Q2 -> weekly closes after that get a P/B
        px = _daily_prices("INTC", date(2026, 7, 23), 260, lambda i: 100.0)
        qs = [{"quarter": "2026Q2", "book_value_per_share": 20.0}]
        s = pb_ratio_series(px, "INTC", qs)
        assert not s.is_empty()
        # 100 / 20 = 5.0
        assert s.sort("period_start").tail(1).row(0, named=True)["value"] == pytest.approx(5.0)

    def test_pb_empty_without_bvps(self):
        px = _daily_prices("INTC", date(2026, 7, 23), 260, lambda i: 100.0)
        qs = [{"quarter": "2026Q2", "book_value_per_share": None}]
        assert pb_ratio_series(px, "INTC", qs).is_empty()


def _run(tmp_path, price_df, quarters=None, **kwargs):
    store = _store(tmp_path)
    _seed_fundamentals(store, quarters)
    with patch(
        "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
        return_value=price_df,
    ):
        return run_weekly_update(
            store=store, as_of=date(2026, 7, 24), **kwargs
        ), store


class TestPipeline:
    def test_current_state_no_ladder_no_thesis_fire(self, tmp_path):
        # INTC ~$100 (above every ladder tranche), foundry narrowing, dcai
        # accelerating -> no buy ladder, no thesis sell fires.
        px = _daily_prices("INTC", date(2026, 7, 24), 260, lambda i: 100.0)
        result, store = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        for rid in ("intc_buy_ladder_t1", "intc_buy_ladder_t2", "intc_buy_ladder_t3"):
            assert not by_id[rid].fired, rid
        assert not by_id["intc_sell_foundry_loss_widen"].fired  # -2.4 -> -2.089 narrowing
        assert not by_id["intc_sell_dcai_decel_2q"].fired        # +22 -> +59 rising
        assert by_id["intc_sell_amd_server_share_50"].status == "ok"  # 46.2 < 50
        # P/B ~4.6 -> neither pb rule fires (2.0 < 4.6 < 5.0)
        assert not by_id["intc_buy_pb_below_2"].fired
        assert not by_id["intc_sell_pb_above_5"].fired
        assert store.load_alerts()["rules"]

    def test_ladder_t1_fires_at_70(self, tmp_path):
        px = _daily_prices("INTC", date(2026, 7, 24), 260,
                           lambda i: 100.0 if i < 250 else 70.0)
        result, _ = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["intc_buy_ladder_t1"].fired       # 70 <= 72
        assert not by_id["intc_buy_ladder_t2"].fired    # 70 > 55

    def test_pb_above_5_fires_when_rich(self, tmp_path):
        # price 120 with BVPS 22.18 -> P/B 5.4 >= 5.0
        px = _daily_prices("INTC", date(2026, 7, 24), 260,
                           lambda i: 90.0 if i < 250 else 120.0)
        result, _ = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["intc_sell_pb_above_5"].fired

    def test_pb_below_2_fires_when_cheap(self, tmp_path):
        px = _daily_prices("INTC", date(2026, 7, 24), 260,
                           lambda i: 100.0 if i < 250 else 40.0)  # 40/22.18=1.8
        result, _ = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["intc_buy_pb_below_2"].fired

    def test_thesis_sells_fire_on_bad_quarter(self, tmp_path):
        px = _daily_prices("INTC", date(2026, 7, 24), 260, lambda i: 100.0)
        bad = list(DEFAULT_FUNDAMENTALS) + [{
            "quarter": "2026Q3", "foundry_op_loss_billions": -2.6,  # widened vs -2.089
            "dcai_revenue_billions": 6.0, "dcai_yoy_growth_pct": 40.0,  # decel 59->40
            "amd_server_rev_share_pct": 51.0,  # >= 50
            "yield_18a_profitable_pct": 60.0,  # < 70 (Q4 redline sim)
            "book_value_per_share": 22.5, "non_gaap_eps": 0.40,
        }]
        result, _ = _run(tmp_path, px, quarters=bad)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["intc_sell_foundry_loss_widen"].fired
        assert by_id["intc_sell_amd_server_share_50"].fired
        assert by_id["intc_sell_18a_yield_miss"].fired
        # one quarter of decel is not enough (rule needs 2 consecutive)
        assert not by_id["intc_sell_dcai_decel_2q"].fired

    def test_dcai_decel_needs_two_quarters(self, tmp_path):
        # +22 -> +59 -> +40 -> +30 : two consecutive decels (59->40->30)
        px = _daily_prices("INTC", date(2026, 7, 24), 260, lambda i: 100.0)
        qs = list(DEFAULT_FUNDAMENTALS) + [
            {"quarter": "2026Q3", "dcai_yoy_growth_pct": 40.0,
             "book_value_per_share": 22.5},
            {"quarter": "2026Q4", "dcai_yoy_growth_pct": 30.0,
             "book_value_per_share": 23.0},
        ]
        result, _ = _run(tmp_path, px, quarters=qs)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["intc_sell_dcai_decel_2q"].fired
        assert by_id["intc_sell_dcai_decel_2q"].streak == 2

    def test_build_ahead_tripwire(self, tmp_path):
        # prefill Q1 $174M (below), Q2 $293M (above) -> streak 0, not tripped now
        px = _daily_prices("INTC", date(2026, 7, 24), 260, lambda i: 100.0)
        result, _ = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert not by_id["intc_sell_build_ahead_tripwire"].fired  # Q2 293 > 250
        # two consecutive quarters of negligible external rev -> trips
        stalled = list(DEFAULT_FUNDAMENTALS) + [
            {"quarter": "2026Q3", "foundry_external_rev_millions": 180.0,
             "book_value_per_share": 22.5},
            {"quarter": "2026Q4", "foundry_external_rev_millions": 210.0,
             "book_value_per_share": 23.0},
        ]
        result2, _ = _run(tmp_path, px, quarters=stalled)
        by_id2 = {r.rule_id: r for r in result2.rule_results}
        assert by_id2["intc_sell_build_ahead_tripwire"].fired
        assert by_id2["intc_sell_build_ahead_tripwire"].streak == 2

    def test_14a_commitment_fires_buy(self, tmp_path):
        px = _daily_prices("INTC", date(2026, 7, 24), 260, lambda i: 100.0)
        good = list(DEFAULT_FUNDAMENTALS) + [{
            "quarter": "2026Q3", "foundry_14a_firm_customers": 1.0,
            "foundry_op_loss_billions": -1.8, "dcai_yoy_growth_pct": 60.0,
            "book_value_per_share": 23.0,
        }]
        result, _ = _run(tmp_path, px, quarters=good)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["intc_buy_14a_firm_commitment"].fired

    def test_yield_miss_insufficient_on_prefill(self, tmp_path):
        # prefill leaves yield null -> the yield rules cannot evaluate
        px = _daily_prices("INTC", date(2026, 7, 24), 260, lambda i: 100.0)
        result, _ = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["intc_sell_18a_yield_miss"].status == "insufficient_data"
        assert by_id["intc_buy_18a_yield_hit"].status == "insufficient_data"

    def test_dry_run_writes_nothing(self, tmp_path):
        px = _daily_prices("INTC", date(2026, 7, 24), 260, lambda i: 100.0)
        store = _store(tmp_path)
        _seed_fundamentals(store)
        with patch(
            "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
            return_value=px,
        ):
            run_weekly_update(store=store, as_of=date(2026, 7, 24), dry_run=True)
        assert store.load("prices").is_empty()
        assert store.load_alerts() == {}

    def test_prices_failure_isolated(self, tmp_path):
        store = _store(tmp_path)
        _seed_fundamentals(store)
        with patch(
            "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
            side_effect=RuntimeError("yf down"),
        ):
            result = run_weekly_update(store=store, as_of=date(2026, 7, 24))
        assert result.outcomes[0].status == "failed"
        assert result.total_failure
        by_id = {r.rule_id: r for r in result.rule_results}
        # fundamentals rules still evaluate off stored json
        assert by_id["intc_sell_dcai_decel_2q"].status in ("ok", "insufficient_data")

    def test_provider_routing(self, tmp_path):
        store = _store(tmp_path)
        _seed_fundamentals(store)
        store.save("prices", _daily_prices(
            "INTC", date(2026, 7, 24), 260, lambda i: 100.0
        ).with_columns(pl.lit(dt.datetime(2026, 7, 24)).alias("collected_at")))
        provider = build_intc_series_provider(store)
        assert not provider(
            "intc.price_weekly_close", {"ticker": "INTC"}
        ).is_empty()
        assert not provider("intc.pct_vs_200dma", {"ticker": "INTC"}).is_empty()
        assert not provider("intc.pb_ratio", {"ticker": "INTC"}).is_empty()
        assert not provider(
            "intc.fundamental", {"field": "dcai_yoy_growth_pct"}
        ).is_empty()
        assert provider("unknown.metric", {}).is_empty()
