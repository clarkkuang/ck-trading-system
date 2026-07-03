"""Tests for MonitoringStore."""

import datetime as dt
import json
from datetime import date

import polars as pl

from ck_trading.monitoring.rules import EvalResult
from ck_trading.monitoring.store import MonitoringStore

NOW = dt.datetime(2026, 7, 1)


def _tokens_df(day: date, model: str, tokens: int) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [day], "model_id": [model], "org": ["x"], "bloc": ["chinese"],
        "tokens_total": [tokens], "rank": [1], "scope": ["all"],
        "source": ["api"], "collected_at": [NOW],
    })


class TestParquet:
    def test_round_trip(self, tmp_path):
        store = MonitoringStore(tmp_path)
        df = _tokens_df(date(2026, 6, 29), "a/b", 100)
        store.save("openrouter_daily_tokens", df)
        loaded = store.load("openrouter_daily_tokens")
        assert loaded.height == 1
        assert loaded["tokens_total"][0] == 100

    def test_dedupe_keeps_last(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.save("openrouter_daily_tokens", _tokens_df(date(2026, 6, 29), "a/b", 100))
        store.save("openrouter_daily_tokens", _tokens_df(date(2026, 6, 29), "a/b", 999))
        loaded = store.load("openrouter_daily_tokens")
        assert loaded.height == 1
        assert loaded["tokens_total"][0] == 999

    def test_different_keys_accumulate(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.save("openrouter_daily_tokens", _tokens_df(date(2026, 6, 29), "a/b", 1))
        store.save("openrouter_daily_tokens", _tokens_df(date(2026, 6, 30), "a/b", 2))
        assert store.load("openrouter_daily_tokens").height == 2

    def test_empty_save_noop(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.save("openrouter_daily_tokens", pl.DataFrame())
        assert store.load("openrouter_daily_tokens").is_empty()

    def test_overwrite_replaces(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.overwrite("weekly_bloc_share", pl.DataFrame({"a": [1, 2]}))
        store.overwrite("weekly_bloc_share", pl.DataFrame({"a": [9]}))
        assert store.load("weekly_bloc_share").height == 1

    def test_load_missing_returns_empty(self, tmp_path):
        assert MonitoringStore(tmp_path).load("nope").is_empty()


def _result(rule_id="r1", status="ok", period="2026-W26", value=0.3) -> EvalResult:
    return EvalResult(
        rule_id=rule_id, status=status, metric_value=value, threshold=0.35,
        period_key=period,
        first_period_key=period if status == "triggered" else None,
        streak=8 if status == "triggered" else 1, required=8,
        action_label="Bear", severity="trigger",
    )


class TestAlerts:
    def test_round_trip_and_deterministic(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.save_alerts([_result()])
        raw1 = store.alerts_path.read_text()
        state = store.load_alerts()
        assert state["rules"][0]["rule_id"] == "r1"
        assert raw1.endswith("\n")
        # same-period re-save with same inputs -> identical rules content
        store.save_alerts([_result()])
        state2 = store.load_alerts()
        assert state2["rules"][0]["history"] == state["rules"][0]["history"]

    def test_episode_lifecycle(self, tmp_path):
        store = MonitoringStore(tmp_path)
        # week 1: fires
        store.save_alerts([_result(status="triggered", period="2026-W26", value=0.4)])
        s1 = store.load_alerts()["rules"][0]
        assert s1["status"] == "triggered"
        assert s1["episode_started"] == "2026-W26"
        started_at = s1["triggered_at"]

        # week 2: still true -> same episode start
        store.save_alerts([_result(status="triggered", period="2026-W27", value=0.41)])
        s2 = store.load_alerts()["rules"][0]
        assert s2["episode_started"] == "2026-W26"
        assert s2["triggered_at"] == started_at
        assert s2["period_key"] == "2026-W27"

        # week 3: back below -> resolved
        store.save_alerts([_result(status="ok", period="2026-W28", value=0.3)])
        s3 = store.load_alerts()["rules"][0]
        assert s3["status"] == "ok"
        assert s3["resolved_at"] is not None
        assert s3["episode_started"] is None

        # week 4: fires again -> NEW episode
        store.save_alerts([_result(status="triggered", period="2026-W29", value=0.5)])
        s4 = store.load_alerts()["rules"][0]
        assert s4["episode_started"] == "2026-W29"

    def test_same_period_rerun_replaces_history_point(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.save_alerts([_result(period="2026-W26", value=0.30)])
        store.save_alerts([_result(period="2026-W26", value=0.31)])
        hist = store.load_alerts()["rules"][0]["history"]
        assert len(hist) == 1
        assert hist[0]["value"] == 0.31

    def test_history_capped_at_52(self, tmp_path):
        store = MonitoringStore(tmp_path)
        for i in range(60):
            store.save_alerts([_result(period=f"2026-W{i:02d}", value=0.1)])
        hist = store.load_alerts()["rules"][0]["history"]
        assert len(hist) == 52
        assert hist[-1]["period_key"] == "2026-W59"

    def test_corrupt_file_returns_empty(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.alerts_path.write_text("{not json")
        assert store.load_alerts() == {}


class TestChecklist:
    def test_seeds_defaults(self, tmp_path):
        store = MonitoringStore(tmp_path)
        items = store.load_checklist()
        assert len(items) >= 5
        ids = {i["id"] for i in items}
        assert "menlo_report" in ids
        assert all(i["last_checked"] is None for i in items)
        # seeding does NOT write the file (dashboard is the only writer)
        assert not store.checklist_path.exists()

    def test_save_and_reload(self, tmp_path):
        store = MonitoringStore(tmp_path)
        items = store.load_checklist()
        items[0]["last_checked"] = "2026-07-01"
        items[0]["notes"] = "Anthropic still #1 at 34%"
        store.save_checklist(items)
        again = store.load_checklist()
        target = next(i for i in again if i["id"] == items[0]["id"])
        assert target["last_checked"] == "2026-07-01"
