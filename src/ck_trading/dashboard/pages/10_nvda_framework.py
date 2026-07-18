"""NVDA framework page — three-layer buy/sell signal monitor.

Spec: docs/nvda_analysis_report.md. Layers: technical (calibrated dip +
200dma), valuation (forward P/E from quarterly EPS entry), thesis
(quarterly fundamentals). Verdict banner aggregates active buy vs sell
signals.
"""

from __future__ import annotations

import datetime as dt
from datetime import date

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ck_trading.dashboard.data_cache import (
    load_nvda_alerts,
    load_nvda_json,
    load_nvda_monitoring,
)
from ck_trading.monitoring import nvda_metrics
from ck_trading.monitoring.nvda_config import (
    DEFAULT_SCENARIOS,
    DIP_BUY_THRESHOLD,
    NVDA_RULES,
    WATCH_TICKERS,
)
from ck_trading.monitoring.nvda_pipeline import (
    build_nvda_series_provider,
    nvda_store,
    run_weekly_update,
)
from ck_trading.monitoring.quarters import weighted_fair_value

st.set_page_config(page_title="NVDA Framework", page_icon="🎛️", layout="wide")
st.title("NVDA 投资框架监控 — 买点与卖点")
st.caption(
    "**中性论点评估**(阈值对称,不看持仓成本)。三层信号: ①技术(周度自动: "
    "回撤≥12.7%且趋势内=买 / 连续4周破200日线=卖) ②估值(季录EPS,周算 "
    "forward P/E: <25x买 / >45x卖) ③论点(季度录入: DC减速/指引miss/ASIC份额"
    "/毛利率)。财季标签按**财季结束日所在日历季**(FY27Q2→录成2026Q3)。"
    "规格书: docs/nvda_analysis_report.md"
)

SCENARIO_COLORS = {
    "bull": "#16a34a",
    "base": "#2563eb",
    "mild_bear": "#d97706",
    "structural_bear": "#dc2626",
}

_store = nvda_store()

# --------------------------------------------------------------------------
# Refresh row
# --------------------------------------------------------------------------
refresh_col, status_col = st.columns([1, 4])
with refresh_col:
    run_clicked = st.button(
        "🔄 Run collection now", type="primary", use_container_width=True,
        help="拉最新日线并重评 8 条规则(约 10 秒)",
    )
with status_col:
    if st.session_state.get("nvda_run_summary"):
        st.text(st.session_state["nvda_run_summary"])

if run_clicked:
    with st.spinner("Collecting prices + evaluating rules…"):
        result = run_weekly_update()
    st.session_state["nvda_run_summary"] = result.summary()
    load_nvda_monitoring.clear()
    load_nvda_alerts.clear()
    load_nvda_json.clear()
    st.rerun()

alerts = load_nvda_alerts()
prices = load_nvda_monitoring("prices")
scenarios_doc = load_nvda_json("scenarios")
scenarios = scenarios_doc.get("scenarios") or [dict(s) for s in DEFAULT_SCENARIOS]
fund_doc = load_nvda_json("fundamentals")
quarters = fund_doc.get("quarters", [])

# ---- derived header values -------------------------------------------------
indicators = (
    nvda_metrics.daily_indicators(prices, "NVDA")
    if not prices.is_empty() else pl.DataFrame()
)
latest_ind = (
    indicators.tail(1).row(0, named=True) if not indicators.is_empty() else None
)
fair = weighted_fair_value(scenarios)
eps_info = nvda_metrics.latest_forward_eps(quarters)

# Row 1: price / fair / distance / P/E
m1, m2, m3, m4 = st.columns(4)
if latest_ind:
    m1.metric("NVDA 最新收盘", f"${latest_ind['close']:.2f}",
              delta=str(latest_ind["date"]), delta_color="off")
    m3.metric("现价 vs 公允", f"{latest_ind['close'] / fair - 1:+.1%}",
              delta="低于公允=市场更悲观" if latest_ind["close"] < fair else "",
              delta_color="off")
    if eps_info:
        pe = latest_ind["close"] / eps_info[1]
        m4.metric("forward P/E", f"{pe:.1f}x",
                  delta=f"EPS ${eps_info[1]:.2f} ({eps_info[0]})",
                  delta_color="off")
    else:
        m4.metric("forward P/E", "—", delta="待录入前瞻EPS", delta_color="off")
else:
    m1.metric("NVDA 最新收盘", "—")
    m3.metric("现价 vs 公允", "—")
    m4.metric("forward P/E", "—")
m2.metric("概率加权公允价", f"${fair:.2f}",
          delta="随情景编辑实时重算", delta_color="off")

# Row 2: technical layer readout
if latest_ind and latest_ind.get("dip_pct") is not None:
    t1, t2, t3 = st.columns(3)
    t1.metric("距60日高回撤", f"{latest_ind['dip_pct']:.1%}",
              delta=f"买点阈值 {DIP_BUY_THRESHOLD:.1%}", delta_color="off")
    t2.metric("vs 200日线", f"{latest_ind['pct_vs_200dma']:+.1%}",
              delta="门控开(趋势内)" if latest_ind["pct_vs_200dma"] > 0
              else "门控关(趋势外,回撤不计为买点)",
              delta_color="off")
    t3.metric(
        "60日高 / 200日线",
        f"${latest_ind['high_60d']:.2f} / ${latest_ind['sma_200']:.2f}",
    )

if alerts.get("updated_at"):
    st.caption(f"规则评估: **{alerts['updated_at']}**")
if prices.is_empty():
    st.warning(
        "还没有价格数据 — 先本地跑 `scripts/nvda_monitor_update.py --backfill`"
        "(需要 ~200 交易日历史才能算 200 日线),或点上方按钮。"
    )

# --------------------------------------------------------------------------
# Verdict banner
# --------------------------------------------------------------------------
rule_states = {r["rule_id"]: r for r in alerts.get("rules", [])}
buy_active = [
    (rid, s) for rid, s in rule_states.items()
    if rid.startswith("nvda_buy_") and s.get("status") == "triggered"
]
sell_active = [
    (rid, s) for rid, s in rule_states.items()
    if rid.startswith("nvda_sell_") and s.get("status") == "triggered"
]
insufficient = sum(
    1 for s in rule_states.values() if s.get("status") == "insufficient_data"
)
label_of = {r.rule_id: r.action_label for r in NVDA_RULES}

B, S = len(buy_active), len(sell_active)
if B > 0 and S == 0:
    st.success(
        f"🟢 **该买 — {B} 个买点活跃**: "
        + " · ".join(label_of.get(rid, rid) for rid, _ in buy_active)
    )
elif S > 0 and B == 0:
    st.error(
        f"🔴 **该卖/减 — {S} 个卖点活跃**: "
        + " · ".join(label_of.get(rid, rid) for rid, _ in sell_active)
    )
elif B > 0 and S > 0:
    st.warning(
        f"🟠 **信号冲突({B} 买 / {S} 卖)— 人工裁决,论点层优先于技术层**"
    )
else:
    st.info("⚪ **持有/观望 — 无活跃信号**")
if insufficient:
    st.caption(f"{insufficient}/8 规则数据不足(录入季度数据后自动评估)")

# --------------------------------------------------------------------------
# Buy / sell card groups
# --------------------------------------------------------------------------
def _fmt_value(rule, value):
    if value is None:
        return "—"
    mk = rule.metric_key
    field = rule.dimensions.get("field", "")
    if mk == "nvda.gated_dip":
        return f"{value:.1%}"
    if mk == "nvda.pct_vs_200dma":
        return f"{value:+.1%}"
    if mk == "nvda.forward_pe":
        return f"{value:.1f}x"
    if field in ("dc_qoq_growth_pct", "guide_vs_consensus_pct"):
        return f"{value:+.1f}%"
    if field == "gross_margin_pct":
        return f"{value:.1f}%"
    if field == "asic_server_share_pct":
        return f"{value:.0f}%"
    return f"{value}"


BUY_BADGE = {
    "triggered": ("🟢 买点触发", "success"),
    "ok": ("⚪ 未触发", "info"),
    "insufficient_data": ("🟡 数据不足", "warning"),
}
SELL_BADGE = {
    "triggered": ("🔴 卖点触发", "error"),
    "ok": ("🟢 安全", "success"),
    "insufficient_data": ("🟡 数据不足", "warning"),
}

buy_rules = [r for r in NVDA_RULES if r.rule_id.startswith("nvda_buy_")]
sell_rules = [r for r in NVDA_RULES if r.rule_id.startswith("nvda_sell_")]

st.subheader("买点 Buy signals")
cols = st.columns(len(buy_rules))
for col, rule in zip(cols, buy_rules):
    state = rule_states.get(rule.rule_id, {})
    status = state.get("status", "insufficient_data")
    with col:
        st.metric(
            label=rule.rule_id.removeprefix("nvda_buy_"),
            value=_fmt_value(rule, state.get("metric_value")),
            delta=f"streak {state.get('streak', 0)}/{rule.consecutive_periods}",
            delta_color="off",
        )
        badge, kind = BUY_BADGE.get(status, ("?", "warning"))
        getattr(st, kind)(badge)

st.subheader("卖点 Sell signals")
cols = st.columns(len(sell_rules))
for col, rule in zip(cols, sell_rules):
    state = rule_states.get(rule.rule_id, {})
    status = state.get("status", "insufficient_data")
    with col:
        st.metric(
            label=rule.rule_id.removeprefix("nvda_sell_"),
            value=_fmt_value(rule, state.get("metric_value")),
            delta=f"streak {state.get('streak', 0)}/{rule.consecutive_periods}",
            delta_color="off",
        )
        badge, kind = SELL_BADGE.get(status, ("?", "warning"))
        getattr(st, kind)(badge)

with st.expander("规则定义与告警历史"):
    st.dataframe(
        [
            {
                "rule": r.rule_id, "条件": r.description,
                "动作": r.action_label, "级别": r.severity,
                "状态": rule_states.get(r.rule_id, {}).get("status", "—"),
            }
            for r in NVDA_RULES
        ],
        use_container_width=True, hide_index=True,
    )
    hist_rows = []
    for rid, state in rule_states.items():
        for h in state.get("history", []):
            hist_rows.append({
                "rule": rid, "period": h.get("period_key"),
                "value": h.get("value"), "status": h.get("status"),
            })
    if hist_rows:
        st.dataframe(hist_rows, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Chart 1: NVDA weekly close vs scenario bands (+ SMA200 + 60d high)
# --------------------------------------------------------------------------
st.subheader("NVDA 周收盘 vs 估值情景带")
if prices.is_empty():
    st.info("等待价格数据。")
else:
    closes_w = nvda_metrics.weekly_close_series(prices, "NVDA")
    # last ~3 years for readability
    cutoff = closes_w["period_start"].max() - dt.timedelta(weeks=156)
    closes_w = closes_w.filter(pl.col("period_start") >= cutoff)
    sma_w = nvda_metrics.weekly_sample(indicators, "sma_200").filter(
        pl.col("period_start") >= cutoff
    )
    high_w = nvda_metrics.weekly_sample(indicators, "high_60d").filter(
        pl.col("period_start") >= cutoff
    )

    fig = go.Figure()
    for s in scenarios:
        color = SCENARIO_COLORS.get(s["id"], "#6b7280")
        fig.add_hrect(y0=s["price_low"], y1=s["price_high"],
                      fillcolor=color, opacity=0.07, line_width=0)
        fig.add_hline(
            y=s["implied_price"], line_dash="dot", line_color=color,
            annotation_text=f'${s["implied_price"]:.0f} '
                            f'({s["probability_pct"]:.0f}%)',
            annotation_font_size=10,
        )
    fig.add_hline(y=fair, line_dash="dash", line_color="#111827", line_width=2,
                  annotation_text=f"加权公允 ${fair:.2f}")
    fig.add_trace(go.Scatter(
        x=high_w["period_start"].to_list(), y=high_w["value"].to_list(),
        name="60日高", line=dict(color="#16a34a", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=sma_w["period_start"].to_list(), y=sma_w["value"].to_list(),
        name="SMA200", line=dict(color="#dc2626", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=closes_w["period_start"].to_list(), y=closes_w["value"].to_list(),
        name="NVDA 周收盘", mode="lines",
        line=dict(color="#111827", width=3),
    ))
    fig.update_layout(height=480, yaxis_title="$", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Chart 2: technical minis
# --------------------------------------------------------------------------
if not indicators.is_empty():
    c1, c2 = st.columns(2)
    cutoff2 = indicators["date"].max() - dt.timedelta(weeks=104)
    dip_w = nvda_metrics.weekly_sample(indicators, "dip_pct").filter(
        pl.col("period_start") >= cutoff2
    )
    vs_w = nvda_metrics.weekly_sample(indicators, "pct_vs_200dma").filter(
        pl.col("period_start") >= cutoff2
    )
    with c1:
        f = go.Figure(go.Scatter(
            x=dip_w["period_start"].to_list(),
            y=[v * 100 for v in dip_w["value"].to_list()],
            mode="lines", line=dict(color="#d97706"),
        ))
        f.add_hline(y=DIP_BUY_THRESHOLD * 100, line_dash="dot",
                    line_color="#16a34a",
                    annotation_text=f"买点线 {DIP_BUY_THRESHOLD:.1%}")
        f.update_layout(title="距60日高回撤 %(2年)", height=280,
                        margin=dict(t=40, b=20))
        st.plotly_chart(f, use_container_width=True)
    with c2:
        f = go.Figure(go.Scatter(
            x=vs_w["period_start"].to_list(),
            y=[v * 100 for v in vs_w["value"].to_list()],
            mode="lines", line=dict(color="#2563eb"),
        ))
        f.add_hline(y=0, line_dash="dot", line_color="#dc2626",
                    annotation_text="200日线")
        f.update_layout(title="vs 200日线 %(2年)", height=280,
                        margin=dict(t=40, b=20))
        st.plotly_chart(f, use_container_width=True)

# --------------------------------------------------------------------------
# Chart 3: relative performance
# --------------------------------------------------------------------------
st.subheader("相对表现(归一化 = 100)")
if not prices.is_empty():
    lookback = st.radio("窗口", ["26 周", "52 周", "全部"], horizontal=True,
                        key="nvda_lookback")
    lb = {"26 周": 26, "52 周": 52, "全部": None}[lookback]
    tickers = [t for t, _ in WATCH_TICKERS]
    norm = nvda_metrics.normalized_performance(prices, tickers, lb)
    fig3 = go.Figure()
    for tk in tickers:
        sub = norm.filter(pl.col("ticker") == tk)
        if sub.is_empty():
            continue
        fig3.add_trace(go.Scatter(
            x=sub["week_start"].to_list(), y=sub["norm"].to_list(),
            name=tk, mode="lines",
            line=dict(color="#76b900" if tk == "NVDA" else None,
                      width=3 if tk == "NVDA" else 1.5),
        ))
    fig3.update_layout(height=380, yaxis_title="归一化表现",
                       hovermode="x unified",
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "角色: **SMH**=板块beta(NVDA跑输板块=alpha转负) · **TSM**=供应链景气 · "
        "**AMD/AVGO**=竞争分流(AVGO=定制ASIC proxy)"
    )

# --------------------------------------------------------------------------
# Chart 4: forward P/E
# --------------------------------------------------------------------------
st.subheader("forward P/E(周度自动重算)")
pe_series = nvda_metrics.forward_pe_series(prices, "NVDA", quarters)
if pe_series.is_empty():
    st.info("待录入前瞻 EPS(下方季度表格 forward_eps_consensus 列)。")
else:
    f4 = go.Figure(go.Scatter(
        x=pe_series["period_start"].to_list(),
        y=pe_series["value"].to_list(),
        mode="lines+markers", line=dict(color="#111827"),
    ))
    f4.add_hline(y=25, line_dash="dot", line_color="#16a34a",
                 annotation_text="买点线 25x")
    f4.add_hline(y=45, line_dash="dot", line_color="#dc2626",
                 annotation_text="卖点线 45x")
    f4.update_layout(height=320, yaxis_title="forward P/E",
                     hovermode="x unified")
    st.plotly_chart(f4, use_container_width=True)

# --------------------------------------------------------------------------
# Quarterly fundamentals editor
# --------------------------------------------------------------------------
st.subheader("季度基本面(手动录入 → 规则自动评估)")
st.caption(
    "每季财报后录入(下次 **~2026-08-26,FY27Q2 → 录成 2026Q3**)。"
    "负数合法(指引miss/增速下滑正是信号)。保存后规则立即重评。"
)

fund_rows = [
    {
        "季度": q.get("quarter", ""),
        "DC营收 $B": q.get("dc_revenue_billions"),
        "DC QoQ %": q.get("dc_qoq_growth_pct"),
        "总营收 $B": q.get("total_revenue_billions"),
        "指引vs共识 %": q.get("guide_vs_consensus_pct"),
        "毛利率 %": q.get("gross_margin_pct"),
        "ASIC份额 %": q.get("asic_server_share_pct"),
        "前瞻EPS $": q.get("forward_eps_consensus"),
        "备注": q.get("notes", ""),
    }
    for q in quarters
]
edited_fund = st.data_editor(
    fund_rows or [{
        "季度": "", "DC营收 $B": None, "DC QoQ %": None, "总营收 $B": None,
        "指引vs共识 %": None, "毛利率 %": None, "ASIC份额 %": None,
        "前瞻EPS $": None, "备注": "",
    }],
    num_rows="dynamic", use_container_width=True, hide_index=True,
    column_config={
        "季度": st.column_config.TextColumn(help="格式 2026Q3", width="small"),
        "DC营收 $B": st.column_config.NumberColumn(format="%.1f"),
        "DC QoQ %": st.column_config.NumberColumn(format="%.1f"),
        "总营收 $B": st.column_config.NumberColumn(format="%.1f"),
        "指引vs共识 %": st.column_config.NumberColumn(format="%.1f"),
        "毛利率 %": st.column_config.NumberColumn(format="%.1f"),
        "ASIC份额 %": st.column_config.NumberColumn(format="%.1f"),
        "前瞻EPS $": st.column_config.NumberColumn(format="%.2f"),
        "备注": st.column_config.TextColumn(width="large"),
    },
    key="nvda_fund_editor",
)

if st.button("💾 保存季度数据并重评规则", key="save_nvda_fund"):
    new_quarters = []
    for row in edited_fund:
        label = str(row.get("季度", "")).strip()
        if not label:
            continue
        new_quarters.append({
            "quarter": label,
            "dc_revenue_billions": row.get("DC营收 $B"),
            "dc_qoq_growth_pct": row.get("DC QoQ %"),
            "total_revenue_billions": row.get("总营收 $B"),
            "guide_vs_consensus_pct": row.get("指引vs共识 %"),
            "gross_margin_pct": row.get("毛利率 %"),
            "asic_server_share_pct": row.get("ASIC份额 %"),
            "forward_eps_consensus": row.get("前瞻EPS $"),
            "notes": str(row.get("备注", "") or ""),
        })
    errors = nvda_metrics.validate_fundamentals(new_quarters)
    if errors:
        st.error("校验失败: " + " / ".join(errors))
    else:
        new_quarters.sort(key=lambda q: q["quarter"])
        _store.save_json("fundamentals", {
            "version": 1,
            "updated_at": dt.datetime.now(dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quarters": new_quarters,
        })
        from ck_trading.monitoring.rules import evaluate_rules

        results = evaluate_rules(
            NVDA_RULES, build_nvda_series_provider(nvda_store())
        )
        _store.save_alerts(results)
        load_nvda_json.clear()
        load_nvda_alerts.clear()
        st.toast("已保存并重评规则")
        st.rerun()

# mini trend charts
if quarters:
    from ck_trading.monitoring.quarters import fundamentals_series

    c1, c2, c3 = st.columns(3)
    dcq = fundamentals_series(quarters, "dc_qoq_growth_pct")
    gm = fundamentals_series(quarters, "gross_margin_pct")
    asic = fundamentals_series(quarters, "asic_server_share_pct")
    with c1:
        if not dcq.is_empty():
            f = go.Figure(go.Scatter(
                x=dcq["period_key"].to_list(), y=dcq["value"].to_list(),
                mode="lines+markers", line=dict(color="#76b900"),
            ))
            f.update_layout(title="DC QoQ 增速 %", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c2:
        if not gm.is_empty():
            f = go.Figure(go.Scatter(
                x=gm["period_key"].to_list(), y=gm["value"].to_list(),
                mode="lines+markers", line=dict(color="#2563eb"),
            ))
            f.add_hline(y=70, line_dash="dot", line_color="#dc2626")
            f.update_layout(title="毛利率 %", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c3:
        if not asic.is_empty():
            f = go.Figure(go.Scatter(
                x=asic["period_key"].to_list(), y=asic["value"].to_list(),
                mode="lines+markers", line=dict(color="#d97706"),
            ))
            f.add_hline(y=35, line_dash="dot", line_color="#dc2626")
            f.update_layout(title="ASIC 服务器份额 %", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)

# --------------------------------------------------------------------------
# Scenario probability editor
# --------------------------------------------------------------------------
st.subheader("估值情景与主观概率")
st.caption("概率与价格可编辑;加权公允与上方图表实时联动。保存需概率和=100%。")

from ck_trading.monitoring.quarters import validate_scenarios  # noqa: E402

scen_rows = [
    {
        "情景": s["label"],
        "概率 %": s["probability_pct"],
        "隐含价 $": s["implied_price"],
        "带下沿 $": s["price_low"],
        "带上沿 $": s["price_high"],
        "_id": s["id"],
    }
    for s in scenarios
]
edited_scen = st.data_editor(
    scen_rows, use_container_width=True, hide_index=True,
    disabled=["情景", "_id"],
    column_config={
        "_id": None,
        "概率 %": st.column_config.NumberColumn(format="%.1f", min_value=0,
                                              max_value=100),
        "隐含价 $": st.column_config.NumberColumn(format="%.1f"),
        "带下沿 $": st.column_config.NumberColumn(format="%.1f"),
        "带上沿 $": st.column_config.NumberColumn(format="%.1f"),
    },
    key="nvda_scen_editor",
)
live_scen = [
    {
        "id": row["_id"], "label": row["情景"],
        "probability_pct": row["概率 %"],
        "implied_price": row["隐含价 $"],
        "price_low": row["带下沿 $"],
        "price_high": row["带上沿 $"],
    }
    for row in edited_scen
]
live_errors = validate_scenarios(live_scen)
prob_sum = sum(float(r["概率 %"] or 0) for r in edited_scen)
live_fair = weighted_fair_value(live_scen)

sc1, sc2, sc3 = st.columns(3)
sc1.metric("Σ 概率", f"{prob_sum:.1f}%",
           delta="OK" if abs(prob_sum - 100) < 0.01 else "必须 = 100%",
           delta_color="off")
sc2.metric("编辑后加权公允", f"${live_fair:.2f}")
if latest_ind:
    sc3.metric("现价 vs 编辑后公允",
               f"{latest_ind['close'] / live_fair - 1:+.1%}")
if live_errors:
    st.warning(" / ".join(live_errors))
if st.button("💾 保存情景", key="save_nvda_scen", disabled=bool(live_errors)):
    _store.save_json("scenarios", {
        "version": 1,
        "updated_at": dt.datetime.now(dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenarios": live_scen,
    })
    load_nvda_json.clear()
    st.toast("情景已保存")
    st.rerun()

# --------------------------------------------------------------------------
# Manual checklist
# --------------------------------------------------------------------------
st.subheader("手动检查清单(不可自动化的观测项)")
items = _store.load_checklist()
today = date.today()
cl_rows = []
for it in items:
    last = it.get("last_checked")
    due = True
    if last:
        try:
            due = (today - date.fromisoformat(str(last))).days > it.get(
                "cadence_days", 31
            )
        except ValueError:
            due = True
    cl_rows.append({
        "Due?": "🔔 DUE" if due else "✓",
        "项目": it["label"],
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
        "备注": st.column_config.TextColumn(width="large"),
    },
    key="nvda_checklist_editor",
)
if st.button("💾 保存清单", key="save_nvda_checklist"):
    for it, row in zip(items, edited_cl):
        raw = str(row.get("上次核查", "")).strip()
        it["last_checked"] = raw or None
        it["notes"] = str(row.get("备注", "") or "")
    _store.save_checklist(items)
    st.toast("已保存到 data/monitoring/nvda/checklist.json")
    st.rerun()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("论点速查")
    st.markdown(
        "**三层阈值**\n"
        "- 技术: 回撤≥**12.7%**且趋势内=买 / 连续**4周**破200日线=卖\n"
        "- 估值: P/E<**25x**买 / >**45x**卖\n"
        "- 论点: DC减速**2季** / 指引**<0** / ASIC≥**35%** / 毛利<**70%**\n\n"
        "**关键锚点(2026-07)**\n"
        "- Q1 FY27: 营收$81.6B(+85%), DC$75.2B, 毛利75%\n"
        "- Hyperscaler capex 2026: ~$725B(+77%)\n"
        "- 现金流曲线 Q3'26 交叉; capex/收入分歧46%>电信泡沫32%\n"
        "- ASIC 份额 27.8% → 35% 是质变线\n"
        "- Top4 客户 >50% 收入\n\n"
        "**与页 07 dip 系统的关系**\n"
        "- 本页技术层买点(周度)无反弹过滤; 页 07 另加 5 日反弹确认\n"
        "- 本页触发而页 07 显示 WATCH = 区间已到、时机未确认 — 以本页为准, "
        "页 07 只回答\"现在下还是等反弹\"\n\n"
        "*持仓 1303 股 @ $20.85 — 仅作参考,信号不看成本*"
    )
    st.divider()
    st.markdown("**监控标的**")
    st.dataframe(
        [{"ticker": t, "角色": r} for t, r in WATCH_TICKERS],
        use_container_width=True, hide_index=True,
    )
