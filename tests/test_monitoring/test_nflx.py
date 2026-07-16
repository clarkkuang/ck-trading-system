"""Tests for the NFLX framework monitor (config, metrics, pipeline)."""

import datetime as dt
from datetime import date, timedelta
from unittest.mock import patch

import polars as pl
import pytest

from ck_trading.monitoring.nflx_config import (
    DEFAULT_FUNDAMENTALS,
    DEFAULT_SCENARIOS,
    NFLX_RULES,
)
from ck_trading.monitoring.nflx_metrics import validate_fundamentals
from ck_trading.monitoring.nflx_pipeline import (
    build_nflx_series_provider,
    nflx_store,
    run_weekly_update,
)
from ck_trading.monitoring.quarters import weighted_fair_value
from ck_trading.monitoring.store import MonitoringStore
from ck_trading.monitoring.nflx_config import DEFAULT_CHECKLIST, NFLX_DEDUPE_KEYS


def _store(tmp_path) -> MonitoringStore:
    return MonitoringStore(
        tmp_path, dedupe_keys=NFLX_DEDUPE_KEYS, checklist_seed=DEFAULT_CHECKLIST
    )


def _daily_prices(
    ticker: str, end: date, n: int, closes_fn
) -> pl.DataFrame:
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
        "updated_at": "2026-07-17T00:00:00Z",
        "quarters": list(quarters if quarters is not None else DEFAULT_FUNDAMENTALS),
    })


class TestConfig:
    def test_scenarios_valid_and_weighted_fair(self):
        assert weighted_fair_value(list(DEFAULT_SCENARIOS)) == pytest.approx(81.25)
        probs = sum(s["probability_pct"] for s in DEFAULT_SCENARIOS)
        assert probs == pytest.approx(100.0)

    def test_rules_wellformed_and_prefixed(self):
        ids = [r.rule_id for r in NFLX_RULES]
        assert len(ids) == len(set(ids))
        for r in NFLX_RULES:
            assert r.rule_id.startswith(("nflx_buy_", "nflx_sell_"))

    def test_prefill_fundamentals_validate(self):
        assert validate_fundamentals(list(DEFAULT_FUNDAMENTALS)) == []


class TestValidator:
    def test_negative_growth_allowed(self):
        qs = [{"quarter": "2026Q3", "revenue_growth_pct": -2.0}]
        assert validate_fundamentals(qs) == []

    def test_negative_fcf_rejected(self):
        qs = [{"quarter": "2026Q3", "fcf_billions": -1.0}]
        assert any("fcf" in e for e in validate_fundamentals(qs))

    def test_ads_on_track_binary(self):
        assert validate_fundamentals([{"quarter": "2026Q3", "ads_on_track": 0.5}])
        assert validate_fundamentals([{"quarter": "2026Q3", "ads_on_track": 1.0}]) == []

    def test_margin_range(self):
        assert any(
            "op_margin" in e
            for e in validate_fundamentals(
                [{"quarter": "2026Q3", "op_margin_pct": 105.0}]
            )
        )

    def test_bad_label(self):
        assert validate_fundamentals([{"quarter": "FY27Q1"}])


def _run(tmp_path, price_df, quarters=None, **kwargs):
    store = _store(tmp_path)
    _seed_fundamentals(store, quarters)
    with patch(
        "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
        return_value=price_df,
    ):
        return run_weekly_update(
            store=store, as_of=date(2026, 7, 17), **kwargs
        ), store


class TestPipeline:
    def test_happy_path_ladder_t1_fires_at_68(self, tmp_path):
        # NFLX flat 90 for warmup then drops to 68 (weekly close 68 <= 70)
        px = _daily_prices("NFLX", date(2026, 7, 17), 260,
                           lambda i: 90.0 if i < 250 else 68.0)
        result, store = _run(tmp_path, px)
        assert result.outcomes[0].status == "ok"
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["nflx_buy_ladder_t1"].fired          # 68 <= 70
        assert not by_id["nflx_buy_ladder_t2"].fired      # 68 > 63
        # below the 200dma -> trend repair NOT fired
        assert not by_id["nflx_buy_trend_repair"].fired
        # prefill fundamentals keep all sell rules quiet
        for rid in ("nflx_sell_rev_growth_below_10", "nflx_sell_guide_below_10",
                    "nflx_sell_ads_off_track", "nflx_sell_margin_below_28"):
            assert by_id[rid].status == "ok", rid
        # alerts persisted
        assert store.load_alerts()["rules"]

    def test_ladder_t2_fires_at_62(self, tmp_path):
        px = _daily_prices("NFLX", date(2026, 7, 17), 260,
                           lambda i: 90.0 if i < 250 else 62.0)
        result, _ = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["nflx_buy_ladder_t1"].fired
        assert by_id["nflx_buy_ladder_t2"].fired

    def test_trend_repair_fires_above_200dma(self, tmp_path):
        # rising series: close ends well above its 200d mean
        px = _daily_prices("NFLX", date(2026, 7, 17), 260,
                           lambda i: 60.0 + i * 0.2)  # ends ~112, sma ~86
        result, _ = _run(tmp_path, px)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["nflx_buy_trend_repair"].fired
        assert not by_id["nflx_buy_ladder_t1"].fired  # price ~112 > 70

    def test_sell_rules_fire_on_bad_quarter(self, tmp_path):
        px = _daily_prices("NFLX", date(2026, 7, 17), 260, lambda i: 90.0)
        bad = list(DEFAULT_FUNDAMENTALS) + [{
            "quarter": "2026Q3", "revenue_growth_pct": 9.5,
            "next_q_guide_growth_pct": 8.0, "op_margin_pct": 27.0,
            "ads_on_track": 0.0, "fcf_billions": 2.0,
            "buyback_billions": 3.0, "notes": "synthetic bear quarter",
        }]
        result, _ = _run(tmp_path, px, quarters=bad)
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["nflx_sell_rev_growth_below_10"].fired
        assert by_id["nflx_sell_guide_below_10"].fired
        assert by_id["nflx_sell_ads_off_track"].fired
        assert by_id["nflx_sell_margin_below_28"].fired

    def test_no_fundamentals_insufficient(self, tmp_path):
        store = _store(tmp_path)  # no fundamentals.json
        px = _daily_prices("NFLX", date(2026, 7, 17), 260, lambda i: 90.0)
        with patch(
            "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
            return_value=px,
        ):
            result = run_weekly_update(store=store, as_of=date(2026, 7, 17))
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["nflx_sell_rev_growth_below_10"].status == "insufficient_data"
        assert any(
            o.name == "fundamentals" and o.status == "empty"
            for o in result.outcomes
        )

    def test_dry_run_writes_nothing(self, tmp_path):
        px = _daily_prices("NFLX", date(2026, 7, 17), 260, lambda i: 90.0)
        store = _store(tmp_path)
        with patch(
            "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
            return_value=px,
        ):
            run_weekly_update(store=store, as_of=date(2026, 7, 17), dry_run=True)
        assert store.load("prices").is_empty()
        assert store.load_alerts() == {}

    def test_prices_failure_isolated(self, tmp_path):
        store = _store(tmp_path)
        _seed_fundamentals(store)
        with patch(
            "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
            side_effect=RuntimeError("yf down"),
        ):
            result = run_weekly_update(store=store, as_of=date(2026, 7, 17))
        assert result.outcomes[0].status == "failed"
        assert result.total_failure  # prices is the only collection section
        # fundamentals rules still evaluated off the stored json
        by_id = {r.rule_id: r for r in result.rule_results}
        assert by_id["nflx_sell_rev_growth_below_10"].status == "ok"

    def test_provider_routing(self, tmp_path):
        store = _store(tmp_path)
        _seed_fundamentals(store)
        store.save("prices", _daily_prices(
            "NFLX", date(2026, 7, 17), 260, lambda i: 90.0
        ).with_columns(pl.lit(dt.datetime(2026, 7, 17)).alias("collected_at")))
        provider = build_nflx_series_provider(store)
        assert not provider(
            "nflx.price_weekly_close", {"ticker": "NFLX"}
        ).is_empty()
        assert not provider("nflx.pct_vs_200dma", {"ticker": "NFLX"}).is_empty()
        assert not provider(
            "nflx.fundamental", {"field": "revenue_growth_pct"}
        ).is_empty()
        assert provider("unknown.metric", {}).is_empty()
