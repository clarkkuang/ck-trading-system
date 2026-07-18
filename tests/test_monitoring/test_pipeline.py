"""Tests for the weekly pipeline orchestration."""

import datetime as dt
from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ck_trading.collectors.openrouter import (
    OpenRouterCollector,
    RankingsSchemaError,
)
from ck_trading.collectors.pkg_downloads import PackageDownloadsCollector
from ck_trading.monitoring.pipeline import (
    build_series_provider,
    run_weekly_update,
)
from ck_trading.monitoring.store import MonitoringStore

NOW = dt.datetime(2026, 7, 1)
AS_OF = date(2026, 7, 1)


def _pricing_df() -> pl.DataFrame:
    return pl.DataFrame({
        "snapshot_date": [AS_OF],
        "model_id": ["anthropic/claude-sonnet-5"],
        "org": ["anthropic"], "bloc": ["anthropic"],
        "prompt_usd_per_mtok": [3.0], "completion_usd_per_mtok": [15.0],
        "context_length": [200000], "created_at": [NOW], "collected_at": [NOW],
    })


def _rankings_df() -> pl.DataFrame:
    rows = []
    for i in range(7):
        d = date(2026, 6, 22) + dt.timedelta(days=i)
        rows.append({
            "date": d, "model_id": "anthropic/claude-sonnet-5",
            "org": "anthropic", "bloc": "anthropic", "tokens_total": 1_000_000,
            "rank": 1, "scope": "all", "source": "api", "collected_at": NOW,
        })
    return pl.DataFrame(rows)


def _npm_df() -> pl.DataFrame:
    return pl.DataFrame({
        "registry": ["npm"], "package": ["@anthropic-ai/claude-code"],
        "period_start": [date(2026, 6, 22)], "period_end": [date(2026, 6, 28)],
        "downloads": [10_000_000], "iso_week": ["2026-W26"],
        "collected_at": [NOW],
    })


def _pypi_df() -> pl.DataFrame:
    return pl.DataFrame({
        "registry": ["pypi"], "package": ["anthropic"],
        "period_start": [date(2026, 6, 24)], "period_end": [AS_OF],
        "downloads": [2_000_000], "iso_week": ["2026-W26"],
        "collected_at": [NOW],
    })


@pytest.fixture
def patched_collectors():
    with patch.object(OpenRouterCollector, "collect_pricing",
                      return_value=_pricing_df()) as p, \
         patch.object(OpenRouterCollector, "collect_rankings",
                      return_value=_rankings_df()) as r, \
         patch.object(PackageDownloadsCollector, "collect_npm",
                      return_value=_npm_df()) as n, \
         patch.object(PackageDownloadsCollector, "collect_pypi",
                      return_value=_pypi_df()) as y:
        yield {"pricing": p, "rankings": r, "npm": n, "pypi": y}


def _outcome(result, name):
    return next(o for o in result.outcomes if o.name == name)


class TestPipeline:
    def test_happy_path_with_key(self, tmp_path, patched_collectors):
        store = MonitoringStore(tmp_path)
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store)

        for name in ("pricing", "rankings", "npm", "pypi", "derive", "alerts"):
            assert _outcome(result, name).status in ("ok", "empty"), name
        assert not result.total_failure
        # data persisted
        assert not store.load("openrouter_pricing_snapshots").is_empty()
        assert not store.load("openrouter_daily_tokens").is_empty()
        assert not store.load("pkg_downloads").is_empty()
        assert not store.load("weekly_bloc_share").is_empty()
        assert store.alerts_path.exists()

    def test_no_key_skips_rankings(self, tmp_path, patched_collectors):
        store = MonitoringStore(tmp_path)
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = ""
            result = run_weekly_update(as_of=AS_OF, store=store)
        assert _outcome(result, "rankings").status == "skipped"
        assert _outcome(result, "pricing").status == "ok"
        assert not result.total_failure

    def test_rankings_failure_isolated(self, tmp_path, patched_collectors):
        patched_collectors["rankings"].side_effect = RankingsSchemaError("shape!")
        store = MonitoringStore(tmp_path)
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store)
        assert _outcome(result, "rankings").status == "failed"
        assert _outcome(result, "pricing").status == "ok"
        assert _outcome(result, "npm").status == "ok"
        assert not result.total_failure

    def test_total_failure_flag(self, tmp_path, patched_collectors):
        for m in patched_collectors.values():
            m.side_effect = RuntimeError("network down")
        store = MonitoringStore(tmp_path)
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store)
        assert result.total_failure

    def test_dry_run_writes_nothing(self, tmp_path, patched_collectors):
        store = MonitoringStore(tmp_path)
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store, dry_run=True)
        assert _outcome(result, "pricing").status == "ok"
        assert store.load("openrouter_pricing_snapshots").is_empty()
        assert not store.alerts_path.exists()

    def test_derive_uses_stored_history_when_this_week_fails(
        self, tmp_path, patched_collectors
    ):
        """Rankings fail this run, but prior stored history still derives."""
        store = MonitoringStore(tmp_path)
        store.save("openrouter_daily_tokens", _rankings_df())
        store.save("openrouter_pricing_snapshots", _pricing_df())

        patched_collectors["rankings"].side_effect = RankingsSchemaError("dead")
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store)
        assert _outcome(result, "derive").status == "ok"
        assert not store.load("weekly_bloc_share").is_empty()

    def test_anomalous_day_quarantined_not_saved(self, tmp_path, patched_collectors):
        """A weekly-bucket-sized day is quarantined; normal days still land."""
        store = MonitoringStore(tmp_path)
        store.save("openrouter_daily_tokens", _rankings_df())  # 7d x 1M baseline

        bad_day = date(2026, 6, 29)
        good_day = date(2026, 6, 30)
        new = pl.DataFrame([
            {"date": bad_day, "model_id": "deepseek/deepseek-v4",
             "org": "deepseek", "bloc": "chinese",
             "tokens_total": 20_000_000, "rank": 1, "scope": "all",
             "source": "api", "collected_at": NOW},
            {"date": good_day, "model_id": "anthropic/claude-sonnet-5",
             "org": "anthropic", "bloc": "anthropic",
             "tokens_total": 1_100_000, "rank": 1, "scope": "all",
             "source": "api", "collected_at": NOW},
        ])
        patched_collectors["rankings"].return_value = new
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store)

        outcome = _outcome(result, "rankings")
        assert outcome.status == "ok"
        assert outcome.extra["quarantined_days"] == [
            {"date": "2026-06-29", "scope": "all", "ratio": 20.0}
        ]
        stored = store.load("openrouter_daily_tokens")
        assert good_day in stored["date"].to_list()
        assert stored.filter(
            (pl.col("date") == bad_day) & (pl.col("tokens_total") > 10_000_000)
        ).is_empty()

    def test_all_days_anomalous_fails_section(self, tmp_path, patched_collectors):
        store = MonitoringStore(tmp_path)
        store.save("openrouter_daily_tokens", _rankings_df())

        new = _rankings_df().with_columns(
            (pl.col("tokens_total") * 25).alias("tokens_total")
        )
        patched_collectors["rankings"].return_value = new
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store)

        outcome = _outcome(result, "rankings")
        assert outcome.status == "failed"
        assert "sanity check" in outcome.error
        assert len(outcome.extra["quarantined_days"]) == 7
        # baseline history untouched
        assert store.load("openrouter_daily_tokens")["tokens_total"].max() == 1_000_000

    def test_summary_readable(self, tmp_path, patched_collectors):
        store = MonitoringStore(tmp_path)
        with patch("ck_trading.monitoring.pipeline.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-x"
            result = run_weekly_update(as_of=AS_OF, store=store)
        text = result.summary()
        assert "pricing" in text
        assert "or_cn_dollar_share_gt35_8w" in text


class TestSeriesProvider:
    def test_provider_routes_metric_keys(self, tmp_path):
        store = MonitoringStore(tmp_path)
        store.save("pkg_downloads", _npm_df())
        provider = build_series_provider(store)
        s = provider("pkg.weekly_downloads",
                     {"registry": "npm", "package": "@anthropic-ai/claude-code"})
        assert s.height == 1
        assert s["value"][0] == 10_000_000

    def test_unknown_metric_empty(self, tmp_path):
        provider = build_series_provider(MonitoringStore(tmp_path))
        assert provider("nope", {}).is_empty()
