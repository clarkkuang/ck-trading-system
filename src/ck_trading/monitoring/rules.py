"""Threshold alert engine for the AI model share monitor.

Rules are declarative (`AlertRule`) and evaluated against named metric series
(each a polars DataFrame [period_key, period_start, value] sorted ascending).

Three comparators:

    gt_consecutive
        Latest N periods all above threshold, with inter-period gaps
        <= max_gap_days (tolerates exactly one missed weekly run at the
        default of 14 days; a larger gap breaks the streak — missing weeks
        are never imputed).

    pct_drop_from_trailing_max
        (trailing-window max - latest) / max > threshold.

    decline_streak_with_competitor_growth
        Subject declines week-over-week for N consecutive comparisons AND at
        least one guard series grew over the same span (advisory).

Alert state uses an *episode* model persisted to alerts.json by the store:
fire -> new episode; still-true -> update last_period_key; false -> resolved.
Re-runs within the same period are idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import partial
from typing import Any, Callable, Sequence

import polars as pl


@dataclass(frozen=True)
class AlertRule:
    rule_id: str
    metric_key: str
    dimensions: dict[str, str]
    comparator: str
    threshold: float
    consecutive_periods: int = 1
    window_days: int | None = None
    max_gap_days: int = 14
    inclusive: bool = False  # >= / <= for the threshold-streak comparators
    fallback_dimensions: tuple[dict[str, str], ...] = ()
    guard_dimensions: tuple[dict[str, str], ...] = ()
    action_label: str = ""
    severity: str = "trigger"  # "trigger" | "advisory"
    enabled: bool = True
    description: str = ""


@dataclass
class EvalResult:
    rule_id: str
    status: str  # "ok" | "triggered" | "insufficient_data"
    metric_value: float | None = None
    threshold: float | None = None
    period_key: str | None = None        # latest period evaluated
    first_period_key: str | None = None  # first period of the satisfying streak
    streak: int = 0
    required: int = 1
    action_label: str = ""
    severity: str = "trigger"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return self.status == "triggered"


# A provider returns the series for (metric_key, dimensions) or an empty frame.
SeriesProvider = Callable[[str, dict[str, str]], pl.DataFrame]


def _streak_info(
    series: pl.DataFrame,
    breach: Callable[[float], bool],
    max_gap_days: int,
) -> tuple[int, str | None]:
    """Length of the newest-backwards breach streak and its first period_key.

    Counts from the newest row backwards while (a) breach(value) holds and
    (b) the gap to the previously counted (newer) row <= max_gap_days.
    """
    rows = series.sort("period_start", descending=True).rows(named=True)
    streak = 0
    first_key: str | None = None
    prev_start: date | None = None
    for row in rows:
        if row["value"] is None or not breach(row["value"]):
            break
        if prev_start is not None:
            gap = (prev_start - row["period_start"]).days
            if gap > max_gap_days:
                break
        streak += 1
        first_key = row["period_key"]
        prev_start = row["period_start"]
    return streak, first_key


def _breach_predicate(rule: AlertRule, direction: str) -> Callable[[float], bool]:
    """Threshold predicate: gt / lt, strict unless rule.inclusive."""
    t = rule.threshold
    if direction == "gt":
        return (lambda v: v >= t) if rule.inclusive else (lambda v: v > t)
    return (lambda v: v <= t) if rule.inclusive else (lambda v: v < t)


def _eval_threshold_streak(
    rule: AlertRule, provider: SeriesProvider, *, direction: str
) -> EvalResult:
    dims_chain = (rule.dimensions, *rule.fallback_dimensions)
    last_nonempty: tuple[pl.DataFrame, dict[str, str]] | None = None

    for dims in dims_chain:
        series = provider(rule.metric_key, dims)
        if series.is_empty():
            continue
        last_nonempty = (series, dims)
        if series.height >= rule.consecutive_periods:
            break  # first dims with a full window wins; never mix dims

    if last_nonempty is None:
        return EvalResult(
            rule_id=rule.rule_id, status="insufficient_data",
            threshold=rule.threshold, required=rule.consecutive_periods,
            action_label=rule.action_label, severity=rule.severity,
            details={"reason": "no_data"},
        )

    series, dims_used = last_nonempty
    latest = series.sort("period_start").tail(1).row(0, named=True)
    streak, first_key = _streak_info(
        series, _breach_predicate(rule, direction), rule.max_gap_days
    )
    fallback_used = dims_used is not rule.dimensions

    if series.height < rule.consecutive_periods:
        return EvalResult(
            rule_id=rule.rule_id, status="insufficient_data",
            metric_value=latest["value"], threshold=rule.threshold,
            period_key=latest["period_key"], streak=streak,
            required=rule.consecutive_periods,
            action_label=rule.action_label, severity=rule.severity,
            details={
                "reason": "insufficient_history",
                "periods_available": series.height,
                "dimensions_used": dims_used,
                "fallback_used": fallback_used,
            },
        )

    fired = streak >= rule.consecutive_periods
    return EvalResult(
        rule_id=rule.rule_id,
        status="triggered" if fired else "ok",
        metric_value=latest["value"], threshold=rule.threshold,
        period_key=latest["period_key"],
        first_period_key=first_key if fired else None,
        streak=streak, required=rule.consecutive_periods,
        action_label=rule.action_label, severity=rule.severity,
        details={"dimensions_used": dims_used, "fallback_used": fallback_used},
    )


def _eval_pct_drop(rule: AlertRule, provider: SeriesProvider) -> EvalResult:
    series = provider(rule.metric_key, rule.dimensions)
    if series.is_empty() or series.height < 2:
        return EvalResult(
            rule_id=rule.rule_id, status="insufficient_data",
            threshold=rule.threshold, action_label=rule.action_label,
            severity=rule.severity,
            details={"reason": "no_data" if series.is_empty() else "single_point"},
        )

    series = series.sort("period_start")
    latest = series.tail(1).row(0, named=True)
    window_days = rule.window_days or 365
    cutoff = latest["period_start"] - timedelta(days=window_days)
    window = series.filter(pl.col("period_start") >= cutoff)
    history_days = (latest["period_start"] - series["period_start"].min()).days

    max_row = window.sort("value", descending=True).head(1).row(0, named=True)
    max_val = max_row["value"]
    drop = (max_val - latest["value"]) / max_val if max_val else 0.0
    fired = drop > rule.threshold

    details: dict[str, Any] = {
        "max_value": max_val,
        "max_period": max_row["period_key"],
        "drop_pct": drop,
        "history_days": history_days,
    }
    if "model_id" in series.columns:
        details["max_model_id"] = max_row.get("model_id")
        details["current_model_id"] = latest.get("model_id")

    return EvalResult(
        rule_id=rule.rule_id,
        status="triggered" if fired else "ok",
        metric_value=latest["value"], threshold=rule.threshold,
        period_key=latest["period_key"],
        first_period_key=latest["period_key"] if fired else None,
        streak=1 if fired else 0, required=1,
        action_label=rule.action_label, severity=rule.severity,
        details=details,
    )


def _eval_decline_with_competitor(
    rule: AlertRule, provider: SeriesProvider
) -> EvalResult:
    n = rule.consecutive_periods
    subject = provider(rule.metric_key, rule.dimensions).sort("period_start")
    if subject.height < n + 1:
        return EvalResult(
            rule_id=rule.rule_id, status="insufficient_data",
            threshold=rule.threshold, required=n,
            action_label=rule.action_label, severity=rule.severity,
            details={
                "reason": "insufficient_history",
                "periods_available": subject.height,
                "periods_needed": n + 1,
            },
        )

    tail = subject.tail(n + 1)
    values = tail["value"].to_list()
    keys = tail["period_key"].to_list()
    starts = tail["period_start"].to_list()

    declining = all(values[i + 1] < values[i] for i in range(n))
    # contiguity check on the subject
    contiguous = all(
        (starts[i + 1] - starts[i]).days <= rule.max_gap_days for i in range(n)
    )

    grown: list[dict[str, Any]] = []
    for dims in rule.guard_dimensions:
        g = provider(rule.metric_key, dims).sort("period_start")
        if g.height < n + 1:
            continue
        gv = g.tail(n + 1)["value"].to_list()
        if gv[-1] > gv[0]:
            grown.append({
                "dimensions": dims,
                "growth": (gv[-1] - gv[0]) / gv[0] if gv[0] else None,
            })

    fired = declining and contiguous and bool(grown)
    latest_val = values[-1]
    return EvalResult(
        rule_id=rule.rule_id,
        status="triggered" if fired else "ok",
        metric_value=latest_val, threshold=rule.threshold,
        period_key=keys[-1],
        first_period_key=keys[0] if fired else None,
        streak=n if (declining and contiguous) else 0, required=n,
        action_label=rule.action_label, severity=rule.severity,
        details={
            "declining": declining,
            "contiguous": contiguous,
            "competitors_grown": grown,
        },
    )


def _eval_monotonic_streak(
    rule: AlertRule, provider: SeriesProvider, *, step_holds: Callable[[float, float], bool]
) -> EvalResult:
    """Streak of period-over-period comparisons on the primary series.

    N = rule.consecutive_periods comparisons require N+1 contiguous points.
    step_holds(newer, older) defines the step: rising -> newer > older;
    non-increasing -> newer <= older (flat counts). Gaps > max_gap_days
    break the streak (missing periods never imputed). Streak counted
    newest-backwards so partial streaks report correctly.
    """
    n = rule.consecutive_periods
    series = provider(rule.metric_key, rule.dimensions).sort("period_start")
    if series.height < n + 1:
        return EvalResult(
            rule_id=rule.rule_id, status="insufficient_data",
            threshold=rule.threshold, required=n,
            action_label=rule.action_label, severity=rule.severity,
            details={
                "reason": "insufficient_history",
                "periods_available": series.height,
                "periods_needed": n + 1,
            },
        )

    rows = series.rows(named=True)
    # count qualifying steps from the newest comparison backwards
    streak = 0
    first_key: str | None = None
    for i in range(len(rows) - 1, 0, -1):
        newer, older = rows[i], rows[i - 1]
        if newer["value"] is None or older["value"] is None:
            break
        gap = (newer["period_start"] - older["period_start"]).days
        if gap > rule.max_gap_days:
            break
        if not step_holds(newer["value"], older["value"]):
            break
        streak += 1
        first_key = older["period_key"]
        if streak >= len(rows) - 1:
            break

    fired = streak >= n
    latest = rows[-1]
    return EvalResult(
        rule_id=rule.rule_id,
        status="triggered" if fired else "ok",
        metric_value=latest["value"], threshold=rule.threshold,
        period_key=latest["period_key"],
        first_period_key=first_key if fired else None,
        streak=streak, required=n,
        action_label=rule.action_label, severity=rule.severity,
        details={},
    )


_COMPARATORS: dict[str, Callable[[AlertRule, SeriesProvider], EvalResult]] = {
    "gt_consecutive": partial(_eval_threshold_streak, direction="gt"),
    "lt_consecutive": partial(_eval_threshold_streak, direction="lt"),
    "pct_drop_from_trailing_max": _eval_pct_drop,
    "decline_streak_with_competitor_growth": _eval_decline_with_competitor,
    "rising_streak": partial(
        _eval_monotonic_streak, step_holds=lambda newer, older: newer > older
    ),
    "non_increasing_streak": partial(
        _eval_monotonic_streak, step_holds=lambda newer, older: newer <= older
    ),
}


def evaluate_rules(
    rules: Sequence[AlertRule], provider: SeriesProvider
) -> list[EvalResult]:
    """Evaluate every enabled rule; a broken rule yields insufficient_data
    with the error recorded rather than raising."""
    results: list[EvalResult] = []
    for rule in rules:
        if not rule.enabled:
            continue
        fn = _COMPARATORS.get(rule.comparator)
        if fn is None:
            results.append(EvalResult(
                rule_id=rule.rule_id, status="insufficient_data",
                threshold=rule.threshold, action_label=rule.action_label,
                severity=rule.severity,
                details={"reason": f"unknown_comparator:{rule.comparator}"},
            ))
            continue
        try:
            results.append(fn(rule, provider))
        except Exception as e:  # noqa: BLE001 — one bad rule must not kill the run
            results.append(EvalResult(
                rule_id=rule.rule_id, status="insufficient_data",
                threshold=rule.threshold, action_label=rule.action_label,
                severity=rule.severity,
                details={"reason": "evaluation_error", "error": repr(e)},
            ))
    return results
