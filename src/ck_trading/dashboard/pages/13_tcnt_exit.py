"""Tencent (0700.HK) EXIT page — a liquidation ladder, not an allocation view.

Unlike pages 09-12 this page never renders a buy verdict. The direction is
settled (full exit over 20 quarterly tranches); the page answers only "how
much do I sell this quarter, and has anything broken that means stop waiting
for a better price".

Layout differs accordingly: a progress bar and a band verdict replace the
fair-value banner, and the invalidation rules get their own counter strip
because in a five-year plan they are the load-bearing wall, not a footnote.
"""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ck_trading.dashboard.data_cache import (
    load_tcnt_alerts,
    load_tcnt_json,
    load_tcnt_monitoring,
)
from ck_trading.monitoring import tcnt_metrics
from ck_trading.monitoring.tcnt_config import (
    EXIT_BASELINE_PCT,
    EXIT_TRANCHES,
    PE_CLEAR_ALL,
    PE_CLEAR_HALF,
    PE_DEEP_BRAKE,
    SKIP_BUDGET,
    TCNT_RULES,
    WATCH_TICKERS,
)
from ck_trading.monitoring.tcnt_pipeline import (
    run_weekly_update,
    tcnt_store,
)

SUBJECT = "0700.HK"
KILL_PREFIXES = ("tcnt_kill_", "tcnt_buyback_")

st.set_page_config(page_title="腾讯 清仓监控", page_icon="🚪", layout="wide")
st.title("🚪 腾讯 0700.HK — 清仓路径监控")
st.caption(
    "方向已定:五年内清仓。本页不产生任何买入判定,只回答两件事——"
    "**本季卖多少**,以及**有没有东西坏掉、需要停止等待更好价格**。"
)

_store = tcnt_store()
plan = load_tcnt_json("exit_plan")
fund = load_tcnt_json("fundamentals")
quarters = fund.get("quarters", [])
alerts = load_tcnt_alerts()
prices = load_tcnt_monitoring("prices")

# --------------------------------------------------------------------------
# Refresh
# --------------------------------------------------------------------------
c1, c2 = st.columns([1, 5])
with c1:
    if st.button("🔄 重新评估", key="tcnt_refresh"):
        with st.spinner("拉取 0700.HK 与港股同业…"):
            res = run_weekly_update()
        st.toast(f"{res.title}: {len(res.rule_results)} 条规则")
        load_tcnt_monitoring.clear()
        load_tcnt_alerts.clear()
        load_tcnt_json.clear()
        st.rerun()
with c2:
    if alerts.get("updated_at"):
        st.caption(f"规则最后评估:{alerts['updated_at']}")

# --------------------------------------------------------------------------
# Current multiple + band verdict
# --------------------------------------------------------------------------
pe_series = tcnt_metrics.trailing_pe_series(prices, SUBJECT, quarters)
cur_pe = float(pe_series["value"][-1]) if not pe_series.is_empty() else None
eps_pair = tcnt_metrics.latest_trailing_eps_hkd(quarters)
band = tcnt_metrics.exit_band(cur_pe, pe_series)

spot = None
if not prices.is_empty():
    sub = prices.filter(pl.col("ticker") == SUBJECT).sort("date")
    if not sub.is_empty():
        spot = float(sub["close"][-1])

m1, m2, m3, m4 = st.columns(4)
m1.metric("现价 (HK$)", f"{spot:,.2f}" if spot else "—")
m2.metric(
    "非IFRS 年化 EPS",
    f"HK${eps_pair[1]:,.2f}" if eps_pair else "—",
    help=f"基于 {eps_pair[0]}" if eps_pair else "需先录入季度非IFRS归母净利",
)
m3.metric("当前 P/E", f"{cur_pe:,.2f}x" if cur_pe else "—")
if band.get("low_52w"):
    m4.metric(
        "52周 P/E 区间",
        f"{band['low_52w']:.1f} – {band['high_52w']:.1f}x",
        help="档位边界取自该区间的分位数,每季随数据滚动 —— "
             "固定价位在五年里会自动失效",
    )
else:
    m4.metric("52周 P/E 区间", "数据不足")

# --------------------------------------------------------------------------
# Progress + this quarter's tranche
# --------------------------------------------------------------------------
st.subheader("退出进度")
done = int(plan.get("tranches_executed", 0) or 0)
total = int(plan.get("total_tranches", EXIT_TRANCHES) or EXIT_TRANCHES)
sold_pct = float(plan.get("pct_sold_to_date", 0.0) or 0.0)
st.progress(min(done / total, 1.0) if total else 0.0,
            text=f"已执行 {done}/{total} 档 — 累计卖出 {sold_pct:.1f}%")

p1, p2, p3 = st.columns(3)
tranche = band.get("tranche_pct")
p1.metric(
    "本档判定",
    band.get("band", "—"),
    delta=f"{tranche:.1f}%" if tranche is not None else None,
    delta_color="off",
    help=band.get("note", "档位由当前 P/E 在 52 周区间中的分位决定"),
)
used = int(plan.get("skip_budget_used", 0) or 0)
budget = int(plan.get("skip_budget_total", SKIP_BUDGET) or SKIP_BUDGET)
p2.metric("跳过预算", f"{budget - used} / {budget} 剩余",
          help="仅在 P/E ≤10x 时可用。用完后深度刹车失效 —— "
               "这是防止『再等等』变成『永远不卖』的机制")
gate = plan.get("gate", {}) or {}
if plan.get("first_tranche_deferred"):
    p3.metric("首档闸门", "已推迟一档",
              help=f"P/E ≥ {gate.get('or_price_at_or_above_pe')}x "
                   f"或 {gate.get('hard_deadline')} 之前的首个季末,以先到者为准")
else:
    p3.metric("首档闸门", "已开启")

if plan.get("first_tranche_deferred"):
    trigger_px = None
    if eps_pair and gate.get("or_price_at_or_above_pe"):
        trigger_px = eps_pair[1] * float(gate["or_price_at_or_above_pe"])
    st.info(
        f"⏸️ 首档已按持有人决定推迟一整格。闸门是**规则不是开放式等待**:"
        f"P/E ≥ {gate.get('or_price_at_or_above_pe')}x"
        + (f"(≈ HK${trigger_px:,.0f})" if trigger_px else "")
        + f",或 {gate.get('hard_deadline')} 之前的首个季末 —— 以先到者为准。"
    )

# --------------------------------------------------------------------------
# Invalidation counters — the load-bearing wall of a 5-year plan
# --------------------------------------------------------------------------
st.subheader("论点失效计数器")
st.caption(
    "五年方案的全部合理性建立在『盈利继续增长、AI 在自家 P&L 里变现』上。"
    "任一条触发 = 停止等待价格,全速清仓。"
)
rules_by_id = {r["rule_id"]: r for r in alerts.get("rules", [])}
kill_rules = [r for r in TCNT_RULES if r.rule_id.startswith(KILL_PREFIXES)]
cols = st.columns(len(kill_rules) or 1)
for col, rule in zip(cols, kill_rules):
    got = rules_by_id.get(rule.rule_id, {})
    status = got.get("status", "—")
    streak = got.get("streak", 0) or 0
    need = got.get("required", rule.consecutive_periods) or rule.consecutive_periods
    val = got.get("metric_value")
    icon = {"triggered": "🔴", "ok": "🟢", "insufficient_data": "⚪"}.get(status, "⚪")
    label = rule.action_label.split(" — ")[0] if rule.action_label else rule.rule_id
    col.metric(
        f"{icon} {label[:18]}",
        f"{streak}/{need}",
        delta=f"{val:,.1f}" if isinstance(val, (int, float)) else None,
        delta_color="off",
        help=rule.description or rule.action_label,
    )

fired = [r for r in alerts.get("rules", []) if r.get("fired")]
if fired:
    st.error(
        "🔴 **全速清仓触发**:"
        + "、".join(r.get("action_label", r["rule_id"]) for r in fired)
    )

# --------------------------------------------------------------------------
# P/E chart with the absolute rails
# --------------------------------------------------------------------------
st.subheader("P/E 轨道")
if pe_series.is_empty():
    st.info("需要价格与至少一个季度的非IFRS归母净利才能绘制。")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pe_series["period_start"].to_list(), y=pe_series["value"].to_list(),
        mode="lines", name="P/E (非IFRS TTM)", line={"width": 2},
    ))
    for y, label, color in (
        (PE_CLEAR_ALL, f"清空 {PE_CLEAR_ALL:g}x", "#16a34a"),
        (PE_CLEAR_HALF, f"清半 {PE_CLEAR_HALF:g}x", "#65a30d"),
        (PE_DEEP_BRAKE, f"深度刹车 {PE_DEEP_BRAKE:g}x", "#dc2626"),
    ):
        fig.add_hline(y=y, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right")
    if band.get("low_52w"):
        fig.add_hrect(y0=band["low_52w"], y1=band["high_52w"],
                      fillcolor="#93c5fd", opacity=0.15, line_width=0,
                      annotation_text="52周区间", annotation_position="left")
    fig.update_layout(height=360, margin={"l": 40, "r": 40, "t": 20, "b": 30},
                      yaxis_title="P/E (x)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Relative performance vs HK peers
# --------------------------------------------------------------------------
if not prices.is_empty():
    st.subheader("相对表现")
    perf = tcnt_metrics.normalized_performance(
        prices, [t for t, _ in WATCH_TICKERS]
    )
    if not perf.is_empty():
        fig2 = go.Figure()
        for tk in perf["ticker"].unique().to_list():
            sub = perf.filter(pl.col("ticker") == tk).sort("week_start")
            fig2.add_trace(go.Scatter(
                x=sub["week_start"].to_list(), y=sub["norm"].to_list(),
                mode="lines", name=tk,
                line={"width": 3 if tk == SUBJECT else 1.5},
            ))
        fig2.update_layout(height=320, margin={"l": 40, "r": 20, "t": 20, "b": 30},
                           yaxis_title="归一化 = 100")
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------
# Quarterly fundamentals editor
# --------------------------------------------------------------------------
st.subheader("季度基本面录入")
st.caption("录入后倍数轨与失效计数器立即本地重算,不必等 CI。")
FIELD_COLS = {
    "quarter": "季度",
    "non_ifrs_profit_billions": "非IFRS归母(RMB亿)",
    "non_ifrs_profit_yoy_pct": "同比%",
    "marketing_services_yoy_pct": "营销服务同比%",
    "free_cash_flow_billions": "自由现金流(RMB亿)",
    "net_cash_billions": "净现金(RMB亿)",
    "buyback_hkd_billions": "当季回购(HK$亿)",
    "notes": "备注",
}
rows = [{FIELD_COLS[k]: q.get(k) for k in FIELD_COLS} for q in quarters]
edited = st.data_editor(
    rows or [{v: None for v in FIELD_COLS.values()}],
    num_rows="dynamic", use_container_width=True, hide_index=True,
    key="tcnt_fundamentals_editor",
)
if st.button("💾 保存基本面", key="save_tcnt_fundamentals"):
    inv = {v: k for k, v in FIELD_COLS.items()}
    new_q = []
    for row in edited:
        q = {inv[k]: v for k, v in row.items() if k in inv}
        if str(q.get("quarter", "")).strip():
            new_q.append(q)
    errs = tcnt_metrics.validate_fundamentals(new_q)
    if errs:
        st.error("；".join(errs[:5]))
    else:
        _store.save_json("fundamentals", {
            "version": 1,
            "updated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quarters": new_q,
        })
        load_tcnt_json.clear()
        st.toast("已保存")
        st.rerun()

# --------------------------------------------------------------------------
# Exit-plan state
# --------------------------------------------------------------------------
with st.expander("清仓计划状态 (exit_plan.json)"):
    st.json(plan)

# --------------------------------------------------------------------------
# Checklist
# --------------------------------------------------------------------------
st.subheader("手动检查清单")
items = _store.load_checklist()
today = dt.date.today()
cl_rows = []
for it in items:
    last = it.get("last_checked")
    due = True
    if last:
        try:
            due = (today - dt.date.fromisoformat(str(last))).days > it.get(
                "cadence_days", 31
            )
        except ValueError:
            due = True
    cl_rows.append({
        "Due?": "🔔 DUE" if due else "✓",
        "项目": it.get("label", it.get("id", "?")),
        "链接": it.get("url", ""),
        "周期(天)": it.get("cadence_days", 31),
        "上次核查": str(last) if last else "",
        "备注": it.get("notes", ""),
    })
edited_cl = st.data_editor(
    cl_rows, use_container_width=True, hide_index=True,
    column_config={
        "Due?": st.column_config.TextColumn(disabled=True, width="small"),
        "项目": st.column_config.TextColumn(disabled=True, width="large"),
        "链接": st.column_config.LinkColumn(disabled=True),
        "周期(天)": st.column_config.NumberColumn(disabled=True, width="small"),
        "上次核查": st.column_config.TextColumn(help="YYYY-MM-DD"),
    },
    key="tcnt_checklist_editor",
)
if st.button("💾 保存清单", key="save_tcnt_checklist"):
    for it, row in zip(items, edited_cl):
        it["last_checked"] = str(row.get("上次核查", "")).strip() or None
        it["notes"] = str(row.get("备注", "") or "")
    _store.save_checklist(items)
    st.toast("已保存")
    st.rerun()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("腾讯清仓速查")
    st.markdown(
        f"""
**这不是配置框架,是退出框架**
- 无买入规则(有测试锁住)
- 默认动作 = 卖计划内的档,不是"不动"
- 20 档 / 季度 / 5 年,基线每档 {EXIT_BASELINE_PCT:g}%

**档位(52周区间分位)**
| 分位 | 动作 |
|---|---|
| ≤{PE_DEEP_BRAKE:g}x | 跳过(限{SKIP_BUDGET}次) |
| 下1/3 | 减半 |
| 中1/3 | 基线 |
| 上1/3 | ×2 |
| ≥{PE_CLEAR_HALF:g}x | 卖剩余50% |
| ≥{PE_CLEAR_ALL:g}x | 全清 |

**为什么用倍数不用价格**
5年里 EPS 增长会让固定价位自动失效
——HK$620 今天19x、2031年只剩12.3x。

**为什么不用 SOTP 定卖点**
券商 SOTP 给 HK$652-725,市价 HK$447。
这个 46-62% 折价多年从未收敛,
挂在公允价上等于永远卖不掉。

**回购是双向指标**
上半年 HK$244亿(~HK$5亿/日)是你
卖入的买盘;但它花的是资产负债表
(净现金-22%),缩量 = 买盘消失 +
现金流问题确认。
        """
    )
