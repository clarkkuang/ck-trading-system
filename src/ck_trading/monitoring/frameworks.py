"""Cross-system signal reconciliation (pure functions).

Two signal systems can disagree on the same ticker and both be right,
because they answer different questions:

    framework pages (10 NVDA, 11 NFLX)   authorization layer — WHERE:
        thesis + pre-committed levels; NFLX's ladder deliberately ignores
        the trend gate (value entry), NVDA's dip rule has no reversal
        filter (weekly cadence).

    dip signals page (07)                timing layer — WHEN:
        calibrated dip threshold + 200dma trend gate + N-day reversal
        confirmation (anti-falling-knife), for momentum-style adds.

Precedence: tickers with a dedicated framework read the framework verdict
first; the dip system's verdict is timing color, not a veto. This module
holds the pure logic so the dashboard pages and tests share one source
of truth.
"""

from __future__ import annotations

# ticker -> dashboard page (used by page 07 to label the reference)
FRAMEWORK_TICKERS: dict[str, str] = {
    "NVDA": "10_nvda_framework",
    "NFLX": "11_nflx_framework",
}


def buy_sell_summary(alerts: dict) -> dict:
    """Split an alerts.json document into triggered buy/sell rule lists.

    Relies on the monitor-wide rule_id convention <prefix>_buy_* /
    <prefix>_sell_*. Unknown/missing structures yield empty lists.
    """
    rules = alerts.get("rules", []) if isinstance(alerts, dict) else []

    def _triggered(kind: str) -> list[dict]:
        return [
            r for r in rules
            if isinstance(r, dict)
            and f"_{kind}_" in str(r.get("rule_id", ""))
            and r.get("status") == "triggered"
        ]

    return {
        "buy_triggered": _triggered("buy"),
        "sell_triggered": _triggered("sell"),
        "n_rules": len(rules),
        "updated_at": alerts.get("updated_at") if isinstance(alerts, dict) else None,
    }


def reconcile_verdict(
    buy_fired: bool, sell_fired: bool, dip_verdict: str
) -> tuple[str, str]:
    """Combined reading for a framework ticker: (headline, explanation).

    dip_verdict is page 07's per-holding verdict string ("BUY",
    "WATCH (still falling)", "BLOCKED (downtrend)", "NEAR", "HOLD",
    "INSUFFICIENT DATA").
    """
    if sell_fired and buy_fired:
        return ("🟠 信号冲突", "框架买卖点同时触发 — 人工裁决, 论点层优先于技术层")
    if sell_fired:
        return ("🔴 框架失效/卖点", "以框架为准: 停止加仓, 按失效预案重审; dip 信号不适用")
    if buy_fired:
        if dip_verdict == "BUY":
            return ("🟢 双确认", "框架区间 + dip 时机同时成立 — 最强组合")
        if "still falling" in dip_verdict:
            return ("🟢 按预案执行", "框架买区已到; 时机层未确认(仍在创新低)— "
                    "阶梯/分批的意义就是不赌反弹时点, 可按预案直接分批或等反弹")
        if "downtrend" in dip_verdict:
            return ("🟢 按预案执行", "框架买区已到; dip 系统被趋势门拦截 — "
                    "预承诺价位有意无视趋势(价值型入场), 以框架为准")
        return ("🟢 框架触发", "dip 系统未达自身阈值 — 框架票以框架为准")
    if dip_verdict == "BUY":
        return ("🟢 dip 加仓信号", "框架无触发; 常规趋势内回调加仓路径")
    return ("⚪ 无信号", "框架与 dip 系统均未触发")
