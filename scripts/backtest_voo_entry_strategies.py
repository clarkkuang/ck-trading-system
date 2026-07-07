"""Backtest VOO position-building strategies over the past decade.

Question: given $120,000 available TODAY, how should you deploy into VOO?

Strategies:
    lump_sum   — everything on day 1
    dca_12m    — 1/12 every 21 trading days for a year
    dca_36m    — 1/36 every 21 trading days for three years
    dip_5.5    — deploy a 1/12 tranche whenever price is >=5.5% below its
                 trailing all-time-high (the user's calibrated VOO dip
                 threshold), max one tranche per 21 trading days
    dip_10     — same with a 10% threshold
    dip_20     — same with a 20% threshold (crisis-only)
    god_12m    — DCA-12 executed at each month's LOWEST close (perfect
                 intramonth foresight; theoretical upper bound for timing)

Fairness rules:
    * adj_close (dividend-adjusted) => total-return comparison
    * idle cash earns CASH_APY (default 3%/yr, daily accrual); also run 0%
    * dip strategies never force-deploy: cash can sit forever
    * metrics on the WHOLE portfolio (VOO value + cash)

Robustness: besides the full-window run, evaluate every rolling 5-year
window (monthly starts) and report each strategy's win rate vs lump sum.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl

CAPITAL = 120_000.0
TRANCHE_GAP = 21  # trading days between deployments
DAILY_YIELD = lambda apy: (1 + apy) ** (1 / 252) - 1


def load_voo() -> pl.DataFrame:
    df = pl.read_parquet(
        Path(__file__).resolve().parent.parent / "data/prices/us/daily/VOO.parquet"
    ).sort("date")
    df = df.with_columns(
        pl.coalesce(pl.col("adj_close"), pl.col("close")).alias("px")
    ).filter(pl.col("px").is_not_null() & pl.col("px").is_not_nan())
    return df.select("date", "px")


def simulate(prices: list[float], schedule: dict[int, float], cash_apy: float) -> dict:
    """Replay a deployment schedule {day_index: dollars} over the price path.

    Returns final value, max drawdown, deployment stats.
    """
    dy = DAILY_YIELD(cash_apy)
    cash = CAPITAL
    shares = 0.0
    peak = 0.0
    mdd = 0.0
    deployed_days: list[int] = []

    for i, px in enumerate(prices):
        cash *= 1 + dy
        amt = schedule.get(i, 0.0)
        if amt > 0:
            amt = min(amt, cash)
            shares += amt / px
            cash -= amt
            deployed_days.append(i)
        total = cash + shares * px
        peak = max(peak, total)
        mdd = min(mdd, total / peak - 1)

    final = cash + shares * prices[-1]
    years = len(prices) / 252
    return {
        "final": final,
        "cagr": (final / CAPITAL) ** (1 / years) - 1,
        "max_dd": mdd,
        "n_buys": len(deployed_days),
        "pct_deployed": 1 - cash / final if final > 0 else 0,
        "last_buy_day": deployed_days[-1] if deployed_days else None,
    }


def build_schedules(prices: list[float]) -> dict[str, dict[int, float]]:
    n = len(prices)
    sched: dict[str, dict[int, float]] = {}

    sched["lump_sum"] = {0: CAPITAL}

    sched["dca_12m"] = {i * TRANCHE_GAP: CAPITAL / 12 for i in range(12)}
    sched["dca_36m"] = {i * TRANCHE_GAP: CAPITAL / 36 for i in range(36)}

    # dip strategies: tranche = CAPITAL/12, fire when px <= (1-thr)*ATH,
    # cooldown TRANCHE_GAP days, until fully deployed
    for thr, name in [(0.055, "dip_5.5"), (0.10, "dip_10"), (0.20, "dip_20")]:
        s: dict[int, float] = {}
        ath = prices[0]
        remaining = 12
        cooldown = 0
        for i, px in enumerate(prices):
            ath = max(ath, px)
            if cooldown > 0:
                cooldown -= 1
            if remaining > 0 and cooldown == 0 and px <= (1 - thr) * ath:
                s[i] = CAPITAL / 12
                remaining -= 1
                cooldown = TRANCHE_GAP
        sched[name] = s

    # god mode: 12 monthly tranches, each at the month's lowest close
    god: dict[int, float] = {}
    for m in range(12):
        lo, hi = m * TRANCHE_GAP, min((m + 1) * TRANCHE_GAP, n)
        if lo >= n:
            break
        window = prices[lo:hi]
        best = lo + window.index(min(window))
        god[best] = god.get(best, 0) + CAPITAL / 12
    sched["god_12m"] = god

    return sched


def run_window(prices: list[float], cash_apy: float) -> dict[str, dict]:
    return {
        name: simulate(prices, schedule, cash_apy)
        for name, schedule in build_schedules(prices).items()
    }


def main() -> None:
    df = load_voo()
    px = df["px"].to_list()
    dates = df["date"].to_list()
    print(f"VOO 数据: {dates[0]} → {dates[-1]} ({len(px)} 个交易日, 含股息调整)")
    print(f"本金 ${CAPITAL:,.0f} · 闲置现金年化 3%(另附 0% 敏感性)\n")

    # ---- full-window run --------------------------------------------------
    for apy in (0.03, 0.0):
        res = run_window(px, apy)
        print(f"=== 全窗口 10 年 (现金收益 {apy:.0%}) ===")
        print(f"{'策略':<10}{'终值$':>12}{'CAGR':>8}{'最大回撤':>9}{'买入次数':>9}{'完成建仓':>9}")
        for name, r in sorted(res.items(), key=lambda x: -x[1]["final"]):
            done = "是" if r["n_buys"] >= 12 or name == "lump_sum" else f"{r['n_buys']}/12"
            print(
                f"{name:<10}{r['final']:>12,.0f}{r['cagr']:>8.2%}"
                f"{r['max_dd']:>9.1%}{r['n_buys']:>9}{done:>9}"
            )
        print()

    # ---- rolling 5-year windows -------------------------------------------
    win_len = 252 * 5
    starts = range(0, len(px) - win_len, 21)  # monthly starts
    beats: dict[str, int] = {}
    finals: dict[str, list[float]] = {}
    for s in starts:
        window = px[s : s + win_len]
        res = run_window(window, 0.03)
        ls = res["lump_sum"]["final"]
        for name, r in res.items():
            finals.setdefault(name, []).append(r["final"])
            if name != "lump_sum" and r["final"] > ls:
                beats[name] = beats.get(name, 0) + 1

    n_win = len(list(starts))
    print(f"=== 滚动 5 年窗口稳健性 ({n_win} 个窗口, 月度起点) ===")
    print(f"{'策略':<10}{'跑赢满仓概率':>12}{'中位终值$':>12}{'最差窗口$':>12}")
    med = lambda xs: sorted(xs)[len(xs) // 2]
    for name in ["lump_sum", "god_12m", "dca_12m", "dip_5.5", "dip_10", "dca_36m", "dip_20"]:
        f = finals[name]
        rate = beats.get(name, 0) / n_win if name != "lump_sum" else float("nan")
        rate_s = f"{rate:.0%}" if name != "lump_sum" else "基准"
        print(f"{name:<10}{rate_s:>12}{med(f):>12,.0f}{min(f):>12,.0f}")


if __name__ == "__main__":
    main()
