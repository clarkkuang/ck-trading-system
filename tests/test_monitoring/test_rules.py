"""Tests for the threshold alert engine."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.monitoring.rules import AlertRule, evaluate_rules
from ck_trading.monitoring.rules_config import RULES


def _series(values: list[float], start: date = date(2026, 1, 5),
            gap_days: list[int] | None = None) -> pl.DataFrame:
    """Weekly series; gap_days[i] = days between point i-1 and i (default 7)."""
    dates = [start]
    for i in range(1, len(values)):
        step = gap_days[i] if gap_days else 7
        dates.append(dates[-1] + timedelta(days=step))
    return pl.DataFrame({
        "period_key": [f"P{i}" for i in range(len(values))],
        "period_start": dates,
        "value": values,
    })


def _provider(mapping: dict[tuple[str, str], pl.DataFrame]):
    """mapping key = (metric_key, frozenset-ish str of dims)."""
    def provider(metric_key: str, dims: dict[str, str]) -> pl.DataFrame:
        key = (metric_key, tuple(sorted(dims.items())))
        return mapping.get(key, pl.DataFrame(schema={
            "period_key": pl.Utf8, "period_start": pl.Date, "value": pl.Float64,
        }))
    return provider


def _key(metric: str, **dims) -> tuple:
    return (metric, tuple(sorted(dims.items())))


CN_DIMS = {"bloc": "chinese", "scope": "programming"}


def _cn_rule(**overrides) -> AlertRule:
    base = dict(
        rule_id="cn8w", metric_key="m", dimensions=CN_DIMS,
        comparator="gt_consecutive", threshold=0.35, consecutive_periods=8,
        action_label="Bear", severity="trigger",
    )
    base.update(overrides)
    return AlertRule(**base)


class TestGtConsecutive:
    def test_8_weeks_above_fires(self):
        s = _series([0.40] * 8)
        r = evaluate_rules([_cn_rule()], _provider({_key("m", **CN_DIMS): s}))[0]
        assert r.status == "triggered"
        assert r.streak == 8
        assert r.first_period_key == "P0"

    def test_7_of_8_does_not_fire(self):
        s = _series([0.30] + [0.40] * 7)
        r = evaluate_rules([_cn_rule()], _provider({_key("m", **CN_DIMS): s}))[0]
        assert r.status == "ok"
        assert r.streak == 7

    def test_latest_below_threshold_not_fired(self):
        s = _series([0.40] * 7 + [0.30])
        r = evaluate_rules([_cn_rule()], _provider({_key("m", **CN_DIMS): s}))[0]
        assert r.status == "ok"
        assert r.streak == 0

    def test_gap_14d_tolerated(self):
        # one missed week (14d gap) inside an 8-week streak
        gaps = [7, 7, 7, 14, 7, 7, 7]
        s = _series([0.40] * 8, gap_days=[0] + gaps)
        r = evaluate_rules([_cn_rule()], _provider({_key("m", **CN_DIMS): s}))[0]
        assert r.status == "triggered"

    def test_gap_21d_breaks_streak(self):
        gaps = [7, 7, 7, 21, 7, 7, 7]
        s = _series([0.40] * 8, gap_days=[0] + gaps)
        r = evaluate_rules([_cn_rule()], _provider({_key("m", **CN_DIMS): s}))[0]
        assert r.status == "ok"
        assert r.streak == 4  # only the newest 4 count past the break

    def test_insufficient_history(self):
        s = _series([0.40] * 5)
        r = evaluate_rules([_cn_rule()], _provider({_key("m", **CN_DIMS): s}))[0]
        assert r.status == "insufficient_data"
        assert r.details["periods_available"] == 5

    def test_no_data(self):
        r = evaluate_rules([_cn_rule()], _provider({}))[0]
        assert r.status == "insufficient_data"
        assert r.details["reason"] == "no_data"

    def test_fallback_scope_used(self):
        fb = {"bloc": "chinese", "scope": "all"}
        rule = _cn_rule(fallback_dimensions=(fb,))
        # primary scope has only 3 periods; fallback has 8
        mapping = {
            _key("m", **CN_DIMS): _series([0.40] * 3),
            _key("m", **fb): _series([0.40] * 8),
        }
        r = evaluate_rules([rule], _provider(mapping))[0]
        assert r.status == "triggered"
        assert r.details["fallback_used"] is True
        assert r.details["dimensions_used"] == fb

    def test_primary_with_full_window_preferred(self):
        fb = {"bloc": "chinese", "scope": "all"}
        rule = _cn_rule(fallback_dimensions=(fb,))
        mapping = {
            _key("m", **CN_DIMS): _series([0.40] * 8),
            _key("m", **fb): _series([0.99] * 8),
        }
        r = evaluate_rules([rule], _provider(mapping))[0]
        assert r.details["fallback_used"] is False
        assert r.metric_value == pytest.approx(0.40)


class TestPctDrop:
    def _rule(self) -> AlertRule:
        return AlertRule(
            rule_id="price_cut", metric_key="p",
            dimensions={"family": "anthropic/claude-opus"},
            comparator="pct_drop_from_trailing_max", threshold=0.30,
            window_days=365, action_label="Cut GM",
        )

    def test_36pct_drop_fires(self):
        s = _series([75.0, 75.0, 48.0])
        r = evaluate_rules(
            [self._rule()],
            _provider({_key("p", family="anthropic/claude-opus"): s}),
        )[0]
        assert r.status == "triggered"
        assert r.details["drop_pct"] == pytest.approx((75 - 48) / 75)

    def test_20pct_drop_does_not_fire(self):
        s = _series([75.0, 60.0])
        r = evaluate_rules(
            [self._rule()],
            _provider({_key("p", family="anthropic/claude-opus"): s}),
        )[0]
        assert r.status == "ok"

    def test_recovery_resolves(self):
        """Price dipped then recovered — latest vs max under threshold."""
        s = _series([75.0, 48.0, 70.0])
        r = evaluate_rules(
            [self._rule()],
            _provider({_key("p", family="anthropic/claude-opus"): s}),
        )[0]
        assert r.status == "ok"

    def test_old_max_outside_window_ignored(self):
        # max of 100 sits 400 days before latest -> outside 365d window
        s = _series([100.0, 75.0, 74.0], gap_days=[0, 200, 200])
        r = evaluate_rules(
            [self._rule()],
            _provider({_key("p", family="anthropic/claude-opus"): s}),
        )[0]
        # window max = 75 (74/75 is a ~1% drop)
        assert r.status == "ok"
        assert r.details["max_value"] == pytest.approx(75.0)

    def test_single_point_insufficient(self):
        s = _series([75.0])
        r = evaluate_rules(
            [self._rule()],
            _provider({_key("p", family="anthropic/claude-opus"): s}),
        )[0]
        assert r.status == "insufficient_data"


class TestDeclineWithCompetitor:
    SUBJ = {"registry": "npm", "package": "claude-code"}
    COMP = {"registry": "npm", "package": "codex"}

    def _rule(self) -> AlertRule:
        return AlertRule(
            rule_id="npm_decline", metric_key="d", dimensions=self.SUBJ,
            guard_dimensions=(self.COMP,),
            comparator="decline_streak_with_competitor_growth",
            threshold=0.0, consecutive_periods=3, severity="advisory",
        )

    def test_decline_with_growth_fires(self):
        mapping = {
            _key("d", **self.SUBJ): _series([100, 90, 80, 70]),
            _key("d", **self.COMP): _series([10, 12, 15, 20]),
        }
        r = evaluate_rules([self._rule()], _provider(mapping))[0]
        assert r.status == "triggered"
        assert r.details["competitors_grown"]

    def test_decline_without_competitor_growth_no_fire(self):
        mapping = {
            _key("d", **self.SUBJ): _series([100, 90, 80, 70]),
            _key("d", **self.COMP): _series([20, 18, 15, 10]),
        }
        r = evaluate_rules([self._rule()], _provider(mapping))[0]
        assert r.status == "ok"

    def test_no_decline_no_fire(self):
        mapping = {
            _key("d", **self.SUBJ): _series([100, 90, 95, 92]),
            _key("d", **self.COMP): _series([10, 12, 15, 20]),
        }
        r = evaluate_rules([self._rule()], _provider(mapping))[0]
        assert r.status == "ok"

    def test_insufficient_history(self):
        mapping = {_key("d", **self.SUBJ): _series([100, 90])}
        r = evaluate_rules([self._rule()], _provider(mapping))[0]
        assert r.status == "insufficient_data"


class TestEngineRobustness:
    def test_unknown_comparator(self):
        rule = AlertRule(
            rule_id="bad", metric_key="m", dimensions={},
            comparator="nonexistent", threshold=1.0,
        )
        r = evaluate_rules([rule], _provider({}))[0]
        assert r.status == "insufficient_data"
        assert "unknown_comparator" in r.details["reason"]

    def test_disabled_rule_skipped(self):
        rule = AlertRule(
            rule_id="off", metric_key="m", dimensions={},
            comparator="gt_consecutive", threshold=1.0, enabled=False,
        )
        assert evaluate_rules([rule], _provider({})) == []

    def test_provider_exception_contained(self):
        def bad_provider(metric_key, dims):
            raise RuntimeError("boom")
        rule = AlertRule(
            rule_id="r", metric_key="m", dimensions={},
            comparator="gt_consecutive", threshold=1.0,
        )
        r = evaluate_rules([rule], bad_provider)[0]
        assert r.status == "insufficient_data"
        assert r.details["reason"] == "evaluation_error"


class TestConfiguredRules:
    def test_config_loads_and_is_wellformed(self):
        assert len(RULES) == 4
        ids = [r.rule_id for r in RULES]
        assert len(ids) == len(set(ids))
        for rule in RULES:
            assert rule.comparator in {
                "gt_consecutive",
                "pct_drop_from_trailing_max",
                "decline_streak_with_competitor_growth",
            }

    def test_cn_share_rule_encodes_user_trigger_table(self):
        rule = next(r for r in RULES if r.rule_id == "or_cn_dollar_share_gt35_8w")
        assert rule.threshold == 0.35
        assert rule.consecutive_periods == 8
        assert rule.dimensions["scope"] == "programming"
        assert rule.action_label == "Bear 25%→30%"
