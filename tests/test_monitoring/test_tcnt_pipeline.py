"""Tests for the Tencent exit monitor's pipeline and rule wiring."""

from datetime import date, timedelta
from unittest.mock import patch

import polars as pl

from ck_trading.monitoring.store import MonitoringStore
from ck_trading.monitoring.tcnt_config import (
    DEFAULT_CHECKLIST,
    DEFAULT_FUNDAMENTALS,
    TCNT_DEDUPE_KEYS,
    TCNT_RULES,
)
from ck_trading.monitoring.tcnt_pipeline import (
    build_tcnt_series_provider,
    run_weekly_update,
)

TICKER = "0700.HK"


def _store(tmp_path) -> MonitoringStore:
    return MonitoringStore(
        tmp_path, dedupe_keys=TCNT_DEDUPE_KEYS, checklist_seed=DEFAULT_CHECKLIST
    )


def _daily(ticker: str, end: date, n: int, close: float) -> pl.DataFrame:
    rows, d, i = [], end, n - 1
    while i >= 0:
        if d.weekday() < 5:
            rows.append({"ticker": ticker, "date": d, "close": float(close)})
            i -= 1
        d -= timedelta(days=1)
    return pl.DataFrame(rows).sort("date")


def _seed(store, quarters=None, price=446.6, end=date(2026, 8, 14)):
    store.save("prices", _daily(TICKER, end, 40, price))
    store.save_json("fundamentals", {
        "version": 1,
        "updated_at": "2026-08-16T00:00:00Z",
        "quarters": list(quarters if quarters is not None else DEFAULT_FUNDAMENTALS),
    })


def _by_id(results):
    return {r.rule_id: r for r in results}


class TestNoBuyRules:
    def test_ladder_contains_no_buy_rule(self):
        """The whole point of this instance: direction is pre-decided.

        Matched on id tokens, not substrings — "buyback" is a monitored
        metric (the bid you sell into), not a buy signal.
        """
        for r in TCNT_RULES:
            assert "buy" not in r.rule_id.split("_"), r.rule_id
            assert not any(
                w in r.action_label for w in ("买入", "加仓", "建仓")
            ), r.rule_id

    def test_every_rule_is_pacing_or_invalidation(self):
        for r in TCNT_RULES:
            assert (
                r.rule_id.startswith("tcnt_exit_")
                or r.rule_id.startswith("tcnt_brake_")
                or r.rule_id.startswith("tcnt_kill_")
                or r.rule_id.startswith("tcnt_buyback_")
            ), r.rule_id


class TestProvider:
    def test_routes_pe_and_fundamentals(self, tmp_path):
        store = _store(tmp_path)
        _seed(store)
        p = build_tcnt_series_provider(store)
        assert not p("tcnt.forward_pe", {"ticker": TICKER}).is_empty()
        assert not p("tcnt.price_weekly_close", {"ticker": TICKER}).is_empty()
        assert not p(
            "tcnt.fundamental", {"field": "free_cash_flow_billions"}
        ).is_empty()

    def test_unknown_key_is_empty_not_error(self, tmp_path):
        store = _store(tmp_path)
        _seed(store)
        p = build_tcnt_series_provider(store)
        assert p("tcnt.nope", {}).is_empty()


class TestExitRules:
    def _run(self, tmp_path, price, quarters=None):
        store = _store(tmp_path)
        _seed(store, quarters=quarters, price=price)
        return _by_id(run_weekly_update(
            skip_prices=True, dry_run=True, store=store,
            as_of=date(2026, 8, 16),
        ).rule_results)

    def test_spot_fires_nothing(self, tmp_path):
        """HK$446.6 ~ 13.7x: below every accelerator, above the deep brake."""
        got = self._run(tmp_path, 446.6)
        assert got["tcnt_exit_accelerate_upper_band"].status == "ok"
        assert got["tcnt_exit_clear_half_pe23"].status == "ok"
        assert got["tcnt_exit_clear_all_pe26"].status == "ok"
        assert got["tcnt_brake_deep_value_pe10"].status == "ok"

    def test_upper_band_accelerator(self, tmp_path):
        # 18.2x * 32.7 ~= 595
        assert self._run(tmp_path, 600.0)[
            "tcnt_exit_accelerate_upper_band"].status == "triggered"

    def test_clear_half_at_23x(self, tmp_path):
        assert self._run(tmp_path, 23.0 * 32.7)[
            "tcnt_exit_clear_half_pe23"].status == "triggered"

    def test_clear_all_at_26x(self, tmp_path):
        got = self._run(tmp_path, 26.5 * 32.7)
        assert got["tcnt_exit_clear_all_pe26"].status == "triggered"
        assert got["tcnt_exit_clear_all_pe26"].severity == "trigger"

    def test_deep_brake_at_10x(self, tmp_path):
        assert self._run(tmp_path, 9.5 * 32.7)[
            "tcnt_brake_deep_value_pe10"].status == "triggered"


class TestInvalidationRules:
    def _run(self, tmp_path, quarters):
        store = _store(tmp_path)
        _seed(store, quarters=quarters)
        return _by_id(run_weekly_update(
            skip_prices=True, dry_run=True, store=store,
            as_of=date(2026, 8, 16),
        ).rule_results)

    def test_one_negative_fcf_is_not_enough(self, tmp_path):
        """2026Q2 is counter 1 of 3 — a timing question, not structural."""
        got = self._run(tmp_path, DEFAULT_FUNDAMENTALS)
        assert got["tcnt_kill_fcf_negative_3q"].status == "insufficient_data"

    def test_three_negative_fcf_quarters_fire(self, tmp_path):
        qs = [
            {"quarter": "2026Q2", "free_cash_flow_billions": -13.8},
            {"quarter": "2026Q3", "free_cash_flow_billions": -5.0},
            {"quarter": "2026Q4", "free_cash_flow_billions": -2.0},
        ]
        assert self._run(tmp_path, qs)[
            "tcnt_kill_fcf_negative_3q"].status == "triggered"

    def test_fcf_recovery_breaks_the_streak(self, tmp_path):
        qs = [
            {"quarter": "2026Q2", "free_cash_flow_billions": -13.8},
            {"quarter": "2026Q3", "free_cash_flow_billions": 20.0},
            {"quarter": "2026Q4", "free_cash_flow_billions": -2.0},
        ]
        assert self._run(tmp_path, qs)[
            "tcnt_kill_fcf_negative_3q"].status == "ok"

    def test_ads_growth_collapse_fires(self, tmp_path):
        qs = [
            {"quarter": "2026Q3", "marketing_services_yoy_pct": 8.0},
            {"quarter": "2026Q4", "marketing_services_yoy_pct": 6.0},
        ]
        assert self._run(tmp_path, qs)[
            "tcnt_kill_ads_below_10_2q"].status == "triggered"

    def test_ads_growth_intact_at_22(self, tmp_path):
        got = self._run(tmp_path, DEFAULT_FUNDAMENTALS)
        assert got["tcnt_kill_ads_below_10_2q"].status in {
            "ok", "insufficient_data"
        }

    def test_profit_decline_two_quarters_fires(self, tmp_path):
        """The 5-year case rests on ~9% EPS growth; this watches it directly."""
        qs = [
            {"quarter": "2026Q3", "non_ifrs_profit_yoy_pct": -1.0},
            {"quarter": "2026Q4", "non_ifrs_profit_yoy_pct": -4.0},
        ]
        assert self._run(tmp_path, qs)[
            "tcnt_kill_profit_decline_2q"].status == "triggered"

    def test_net_cash_negative_fires(self, tmp_path):
        qs = [{"quarter": "2026Q3", "net_cash_billions": -10.0}]
        assert self._run(tmp_path, qs)[
            "tcnt_kill_net_cash_negative"].status == "triggered"


class TestBuybackRule:
    def _run(self, tmp_path, quarters):
        store = _store(tmp_path)
        _seed(store, quarters=quarters)
        return _by_id(run_weekly_update(
            skip_prices=True, dry_run=True, store=store,
            as_of=date(2026, 8, 16),
        ).rule_results)

    def test_current_pace_is_safe(self, tmp_path):
        got = self._run(tmp_path, DEFAULT_FUNDAMENTALS)
        assert got["tcnt_buyback_cut_2q"].status in {"ok", "insufficient_data"}

    def test_two_quarters_of_cuts_fire(self, tmp_path):
        """Losing the bid AND confirming the cash squeeze at the same time."""
        qs = [
            {"quarter": "2026Q3", "buyback_hkd_billions": 6.0},
            {"quarter": "2026Q4", "buyback_hkd_billions": 4.0},
        ]
        assert self._run(tmp_path, qs)["tcnt_buyback_cut_2q"].status == "triggered"

    def test_single_light_quarter_is_not_enough(self, tmp_path):
        qs = [
            {"quarter": "2026Q3", "buyback_hkd_billions": 6.0},
            {"quarter": "2026Q4", "buyback_hkd_billions": 14.0},
        ]
        assert self._run(tmp_path, qs)["tcnt_buyback_cut_2q"].status == "ok"


class TestPipelineIsolation:
    def test_price_failure_does_not_stop_alerts(self, tmp_path):
        store = _store(tmp_path)
        _seed(store)
        with patch(
            "ck_trading.collectors.us_market.USMarketCollector.collect_prices",
            side_effect=RuntimeError("hk feed down"),
        ):
            res = run_weekly_update(
                dry_run=True, store=store, as_of=date(2026, 8, 16)
            )
        outcomes = {o.name: o.status for o in res.outcomes}
        assert outcomes["prices"] == "failed"
        assert outcomes["alerts"] == "ok"
        assert len(res.rule_results) == len(TCNT_RULES)

    def test_empty_fundamentals_is_reported_not_fatal(self, tmp_path):
        store = _store(tmp_path)
        store.save("prices", _daily(TICKER, date(2026, 8, 14), 40, 446.6))
        res = run_weekly_update(
            skip_prices=True, dry_run=True, store=store, as_of=date(2026, 8, 16)
        )
        outcomes = {o.name: o.status for o in res.outcomes}
        assert outcomes["fundamentals"] == "empty"
        assert outcomes["alerts"] == "ok"

    def test_dry_run_writes_nothing(self, tmp_path):
        store = _store(tmp_path)
        _seed(store)
        run_weekly_update(
            skip_prices=True, dry_run=True, store=store, as_of=date(2026, 8, 16)
        )
        assert not (tmp_path / "alerts.json").exists()
