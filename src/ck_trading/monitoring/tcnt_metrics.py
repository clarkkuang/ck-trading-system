"""Pure metric functions for the Tencent exit monitor.

Two things here do not exist in the other five instances:

    trailing_pe_series   P/E from the weekly close over trailing non-IFRS EPS,
                         which is the multiple the rails are written against.
    exit_band            the percentile band of the TRAILING 52-WEEK P/E range
                         plus the tranche multiplier — the piece that keeps a
                         five-year ladder from decaying into uselessness.

Everything else (weekly sampling, normalisation) is re-exported from the
shared technicals module.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ck_trading.monitoring.quarters import (  # noqa: F401
    EMPTY_SERIES_SCHEMA,
    QUARTER_RE,
    quarter_period_start,
)
from ck_trading.monitoring.tcnt_config import (
    BAND_MULTIPLIERS,
    EXIT_BASELINE_PCT,
    PE_CLEAR_ALL,
    PE_CLEAR_HALF,
    PE_DEEP_BRAKE,
)
from ck_trading.monitoring.technicals import (  # noqa: F401
    normalize_daily,
    normalized_performance,
    weekly_close_series,
    weekly_sample,
)

#: signed — a loss-making quarter and negative FCF are both real signals
_SIGNED_FIELDS = (
    "non_ifrs_profit_billions", "non_ifrs_profit_yoy_pct",
    "marketing_services_yoy_pct", "free_cash_flow_billions",
    "net_cash_billions",
)
_NON_NEGATIVE_FIELDS = ("buyback_hkd_billions",)


def validate_fundamentals(quarters: list[dict]) -> list[str]:
    """Return human-readable problems (empty = valid)."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(quarters):
        label = str(q.get("quarter", "")).strip()
        if not QUARTER_RE.match(label):
            errors.append(f"第{i + 1}行: 季度标签 {label!r} 不符合 YYYYQ1-4 格式")
            continue
        if label in seen:
            errors.append(f"季度 {label} 重复")
        seen.add(label)
        for fld in _SIGNED_FIELDS + _NON_NEGATIVE_FIELDS:
            val = q.get(fld)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                errors.append(f"{label}: {fld} 不是数字 ({val!r})")
                continue
            if fld in _NON_NEGATIVE_FIELDS and v < 0:
                errors.append(f"{label}: {fld} 为负数 ({v})")
    return errors


def latest_trailing_eps_hkd(
    quarters: list[dict], fx_rmb_to_hkd: float = 1.09, shares_billions: float = 9.126
) -> tuple[str, float] | None:
    """Annualised non-IFRS EPS in HKD from the most recent quarter with data.

    Tencent reports in RMB and trades in HKD, so the multiple is only
    meaningful after the conversion. Annualising a single quarter is crude but
    matches how the rails were calibrated; once four quarters are on file the
    dashboard sums them instead.
    """
    dated = [
        q for q in quarters
        if q.get("non_ifrs_profit_billions") is not None
        and QUARTER_RE.match(str(q.get("quarter", "")))
    ]
    if not dated:
        return None
    dated.sort(key=lambda q: str(q["quarter"]))
    recent = dated[-4:]
    if len(recent) == 4:
        total_rmb = sum(float(q["non_ifrs_profit_billions"]) for q in recent)
    else:
        total_rmb = float(recent[-1]["non_ifrs_profit_billions"]) * 4
    eps_hkd = total_rmb * fx_rmb_to_hkd / shares_billions
    return str(dated[-1]["quarter"]), eps_hkd


def trailing_pe_series(
    prices: pl.DataFrame,
    ticker: str,
    quarters: list[dict],
    fx_rmb_to_hkd: float = 1.09,
    shares_billions: float = 9.126,
) -> pl.DataFrame:
    """Weekly close / trailing annualised non-IFRS EPS, as-of joined.

    Weeks preceding the first usable quarter are dropped rather than valued
    against a fabricated EPS.
    """
    if prices is None or prices.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    weekly = weekly_close_series(prices, ticker)
    if weekly.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)

    rows = []
    for q in quarters:
        label = str(q.get("quarter", "")).strip()
        val = q.get("non_ifrs_profit_billions")
        if not QUARTER_RE.match(label) or val is None:
            continue
        try:
            eps = float(val) * 4 * fx_rmb_to_hkd / shares_billions
        except (TypeError, ValueError):
            continue
        if eps <= 0:
            continue
        rows.append({"period_start": quarter_period_start(label), "eps": eps})
    if not rows:
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)

    eps_df = pl.DataFrame(rows).sort("period_start")
    joined = (
        weekly.sort("period_start")
        .join_asof(eps_df, on="period_start", strategy="backward")
        .drop_nulls("eps")
    )
    if joined.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    return joined.select(
        "period_key", "period_start", (pl.col("value") / pl.col("eps")).alias("value")
    )


def exit_band(current_pe: float, pe_series: pl.DataFrame | None) -> dict:
    """Which exit band the current multiple sits in, and the tranche size.

    The band edges are percentiles of the TRAILING 52-WEEK P/E range rather
    than fixed numbers, so a five-year ladder keeps meaning as earnings grow.
    Absolute rails (deep brake, clear-half, clear-all) override the bands.
    """
    if current_pe is None or current_pe <= 0:
        return {"band": "无数据", "multiplier": None, "tranche_pct": None,
                "low_52w": None, "high_52w": None}

    if current_pe >= PE_CLEAR_ALL:
        return {"band": f"清空 (≥{PE_CLEAR_ALL:g}x)", "multiplier": None,
                "tranche_pct": 100.0, "low_52w": None, "high_52w": None,
                "note": "剩余全部清空"}
    if current_pe >= PE_CLEAR_HALF:
        return {"band": f"清半 (≥{PE_CLEAR_HALF:g}x)", "multiplier": None,
                "tranche_pct": None, "low_52w": None, "high_52w": None,
                "note": "卖出剩余仓位的 50%"}
    if current_pe <= PE_DEEP_BRAKE:
        return {"band": f"深度刹车 (≤{PE_DEEP_BRAKE:g}x)", "multiplier": 0.0,
                "tranche_pct": 0.0, "low_52w": None, "high_52w": None,
                "note": "可跳过本档 — 消耗跳过预算"}

    lo = hi = None
    if pe_series is not None and not pe_series.is_empty():
        tail = pe_series.tail(52)
        lo, hi = float(tail["value"].min()), float(tail["value"].max())
    if lo is None or hi is None or hi <= lo:
        return {"band": "基线 (区间不足)", "multiplier": 1.0,
                "tranche_pct": EXIT_BASELINE_PCT, "low_52w": lo, "high_52w": hi}

    pct = (current_pe - lo) / (hi - lo)
    for label, upper, mult in BAND_MULTIPLIERS:
        if pct <= upper:
            return {"band": label, "multiplier": mult,
                    "tranche_pct": EXIT_BASELINE_PCT * mult,
                    "low_52w": lo, "high_52w": hi, "percentile": pct}
    label, _, mult = BAND_MULTIPLIERS[-1]
    return {"band": label, "multiplier": mult,
            "tranche_pct": EXIT_BASELINE_PCT * mult,
            "low_52w": lo, "high_52w": hi, "percentile": pct}


def relative_strength_series(
    prices: pl.DataFrame,
    ticker: str,
    benchmark: str,
    window_weeks: int = 13,
) -> pl.DataFrame:
    """Rolling excess return of `ticker` over `benchmark`, in percentage points.

    Answers the only question that should make a liquidation ladder go faster:
    is this the subject breaking, or the whole sector? In Feb-2026 Tencent fell
    14.5% while Hang Seng Tech fell 23% — accelerating on that would have sold
    the bottom of a beta move.

    Both legs are sampled to weekly closes first, so a US-listed benchmark and
    an HK-listed subject line up on ISO weeks despite trading different hours.
    Weeks where either leg is missing are dropped rather than carried forward.
    """
    empty = pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    if prices is None or prices.is_empty() or window_weeks < 1:
        return empty
    sub = weekly_close_series(prices, ticker)
    bmk = weekly_close_series(prices, benchmark)
    if sub.is_empty() or bmk.is_empty():
        return empty

    joined = (
        sub.rename({"value": "sub"})
        .join(bmk.rename({"value": "bmk"}).drop("period_start"), on="period_key")
        .sort("period_start")
    )
    if joined.height <= window_weeks:
        return empty

    joined = joined.with_columns([
        (pl.col("sub") / pl.col("sub").shift(window_weeks) - 1).alias("r_sub"),
        (pl.col("bmk") / pl.col("bmk").shift(window_weeks) - 1).alias("r_bmk"),
    ]).drop_nulls(["r_sub", "r_bmk"])
    if joined.is_empty():
        return empty
    return joined.select(
        "period_key",
        "period_start",
        ((pl.col("r_sub") - pl.col("r_bmk")) * 100).alias("value"),
    )


def buyback_blackout(as_of: date, calendar=None, annual_days=None, interim_days=None) -> dict:
    """Whether `as_of` falls inside a buyback blackout, and what is next.

    Not a threshold rule — a calendar fact none of the metric rules can see.
    Tencent stopped buying on 2026-01-16 ahead of 2026-03-18 annual results
    and the HK$500M/day bid simply disappeared for two months.
    """
    from ck_trading.monitoring.tcnt_config import (
        BLACKOUT_DAYS_ANNUAL,
        BLACKOUT_DAYS_INTERIM,
        RESULTS_CALENDAR,
    )

    calendar = RESULTS_CALENDAR if calendar is None else calendar
    annual_days = BLACKOUT_DAYS_ANNUAL if annual_days is None else annual_days
    interim_days = BLACKOUT_DAYS_INTERIM if interim_days is None else interim_days

    windows = []
    for iso, label in calendar:
        results = date.fromisoformat(iso)
        lead = annual_days if "annual" in label.lower() else interim_days
        windows.append((results - timedelta(days=lead), results, label))
    windows.sort()

    for start, end, label in windows:
        if start <= as_of <= end:
            return {
                "in_blackout": True, "label": label,
                "window_start": start, "results_date": end,
                "days_remaining": (end - as_of).days,
                "note": "回购静默期内 —— 无回购托底; 大额减持应避开此窗口",
            }
    upcoming = [w for w in windows if w[0] > as_of]
    if not upcoming:
        return {"in_blackout": False, "label": None, "next_window_start": None,
                "note": "日历已用尽 — 需补录后续业绩日期"}
    start, end, label = upcoming[0]
    return {
        "in_blackout": False, "label": label,
        "next_window_start": start, "results_date": end,
        "days_until": (start - as_of).days,
        "note": f"下一个静默窗口 {start} 起 ({label})",
    }


def sector_vs_country_spread(
    prices: pl.DataFrame,
    sector: str = "KWEB",
    country: str = "MCHI",
    window_weeks: int = 13,
) -> pl.DataFrame:
    """Excess return of the internet sector over broad China, in points.

    Diagnostic, not a trigger. The 2026-08 decomposition of Tencent's -34%
    found the sector down 21.9% over a year while broad China was down 2.6%
    (MCHI) / 4.7% (FXI) — so the drawdown is an industry problem (the instant-
    retail subsidy war burned >RMB100bn in two quarters; Alibaba's e-commerce
    EBITA fell RMB85.7bn, Meituan swung ~RMB60bn) layered on an AI capex
    cycle, NOT a country discount.

    Worth separating because the two have different half-lives: an industry
    margin war is cyclical, a country risk premium is not. Deliberately not
    wired to a rule — it explains WHY, and inventing an action threshold for
    it without evidence would just add noise to the ladder.
    """
    return relative_strength_series(prices, sector, country, window_weeks)
