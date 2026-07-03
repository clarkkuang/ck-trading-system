"""AT&T vs SpaceX page — telecom threat monitor for the AT&T thesis.

Live view of ck_trading.monitoring.att_pipeline: rule status cards,
T price with valuation scenario bands, peer relative performance,
quarterly fundamentals editor, editable scenario probabilities, and the
manual checklist.
"""

from __future__ import annotations

import datetime as dt
from datetime import date

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ck_trading.dashboard.data_cache import (
    load_att_alerts,
    load_att_json,
    load_att_monitoring,
)
from ck_trading.monitoring import att_metrics
from ck_trading.monitoring.att_config import (
    ATT_RULES,
    DEFAULT_SCENARIOS,
    WATCH_TICKERS,
)
from ck_trading.monitoring.att_pipeline import att_store, run_weekly_update

st.set_page_config(page_title="AT&T vs SpaceX", page_icon="📡", layout="wide")
st.title("AT&T vs SpaceX 电信威胁监控")
st.caption(
    "论点: SpaceX 借 EchoStar 频谱(65 MHz)从供应商升级为潜在直营竞争者,"
    "市场定价的是其**成功概率的边际变化**。路径 A 批发/谈判筹码 ~50-55%,"
    "路径 B 直营移动 ~35-40%。物理约束: 卫星 D2C 容量密度比地面网络低 3-4 "
    "个数量级,仅 <20 人/km² 区域可作主承载 — 郊区渗透有硬顶。"
    "详见 docs/att_spacex_monitoring.md。"
)

SCENARIO_COLORS = {
    "base": "#16a34a",
    "mild_bear": "#d97706",
    "structural_decline": "#dc2626",
    "extreme_tail": "#7f1d1d",
}
STATUS_BADGE = {
    "triggered": ("🔴 TRIGGERED", "error"),
    "ok": ("🟢 OK", "success"),
    "insufficient_data": ("🟡 insufficient data", "warning"),
}

_store = att_store()

# --------------------------------------------------------------------------
# Refresh row
# --------------------------------------------------------------------------
refresh_col, status_col = st.columns([1, 4])
with refresh_col:
    run_clicked = st.button(
        "🔄 Run collection now",
        type="primary",
        use_container_width=True,
        help="拉取最新周收盘并重评所有规则(约 10 秒)",
    )
with status_col:
    if st.session_state.get("att_run_summary"):
        st.text(st.session_state["att_run_summary"])

if run_clicked:
    with st.spinner("Collecting prices + evaluating rules…"):
        result = run_weekly_update()
    st.session_state["att_run_summary"] = result.summary()
    load_att_monitoring.clear()
    load_att_alerts.clear()
    load_att_json.clear()
    st.rerun()

alerts = load_att_alerts()
prices = load_att_monitoring("prices")
scenarios_doc = load_att_json("scenarios")
scenarios = scenarios_doc.get("scenarios") or [dict(s) for s in DEFAULT_SCENARIOS]

# ---- thesis metrics ------------------------------------------------------
latest_t_close = None
latest_t_week = None
if not prices.is_empty():
    t_series = att_metrics.weekly_close_series(prices, "T")
    if not t_series.is_empty():
        last = t_series.tail(1).row(0, named=True)
        latest_t_close = last["value"]
        latest_t_week = last["period_key"]

fair = att_metrics.weighted_fair_value(scenarios)
m1, m2, m3 = st.columns(3)
m1.metric(
    "T 最新周收盘",
    f"${latest_t_close:.2f}" if latest_t_close else "—",
    delta=latest_t_week or "",
    delta_color="off",
)
m2.metric("概率加权公允价", f"${fair:.2f}", delta="随下方情景编辑实时重算",
          delta_color="off")
if latest_t_close:
    dist = latest_t_close / fair - 1
    m3.metric("现价 vs 公允", f"{dist:+.1%}",
              delta="低于公允 = 市场比你更悲观" if dist < 0 else "高于公允",
              delta_color="off")
else:
    m3.metric("现价 vs 公允", "—")

freshness = []
if not prices.is_empty():
    freshness.append(f"价格至 **{prices['week_start'].max()}**")
if alerts.get("updated_at"):
    freshness.append(f"规则评估 **{alerts['updated_at']}**")
if freshness:
    st.caption("数据新鲜度: " + " · ".join(freshness))
if prices.is_empty():
    st.warning(
        "还没有价格数据 — 点上方 **Run collection now**,"
        "或等周一的 GitHub Action。下方编辑器不受影响。"
    )

# --------------------------------------------------------------------------
# Rule status cards
# --------------------------------------------------------------------------
st.subheader("触发规则状态")
rule_states = {r["rule_id"]: r for r in alerts.get("rules", [])}
cols = st.columns(len(ATT_RULES))
for col, rule in zip(cols, ATT_RULES):
    state = rule_states.get(rule.rule_id, {})
    status = state.get("status", "insufficient_data")
    value = state.get("metric_value")
    streak = state.get("streak", 0)
    if "price" in rule.metric_key:
        val_str = f"${value:.2f}" if value is not None else "—"
    elif rule.dimensions.get("field") == "churn_pct":
        val_str = f"{value:.2f}%" if value is not None else "—"
    elif rule.dimensions.get("field") == "convergence_pct":
        val_str = f"{value:.1f}%" if value is not None else "—"
    elif rule.dimensions.get("field") == "fcf_billions":
        val_str = f"${value:.1f}B" if value is not None else "—"
    else:
        val_str = f"{value}" if value is not None else "—"
    with col:
        st.metric(
            label=rule.rule_id.removeprefix("att_"),
            value=val_str,
            delta=f"streak {streak}/{rule.consecutive_periods}",
            delta_color="off",
        )
        badge, kind = STATUS_BADGE.get(status, ("?", "warning"))
        getattr(st, kind)(badge)

with st.expander("规则动作与告警历史"):
    st.dataframe(
        [
            {
                "rule": r.rule_id,
                "条件": r.description,
                "动作": r.action_label,
                "级别": r.severity,
                "状态": rule_states.get(r.rule_id, {}).get("status", "—"),
            }
            for r in ATT_RULES
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
# Chart 1: T price with scenario bands
# --------------------------------------------------------------------------
st.subheader("T 周收盘 vs 估值情景带")
if prices.is_empty():
    st.info("等待首次价格采集。")
else:
    t_series = att_metrics.weekly_close_series(prices, "T")
    fig = go.Figure()
    for s in scenarios:
        color = SCENARIO_COLORS.get(s["id"], "#6b7280")
        fig.add_hrect(
            y0=s["price_low"], y1=s["price_high"],
            fillcolor=color, opacity=0.08, line_width=0,
        )
        fig.add_hline(
            y=s["implied_price"], line_dash="dot", line_color=color,
            annotation_text=f'{s["label"][:10]} ${s["implied_price"]:.0f} '
                            f'({s["probability_pct"]:.0f}%)',
            annotation_font_size=10,
        )
    fig.add_hline(
        y=fair, line_dash="dash", line_color="#2563eb", line_width=2,
        annotation_text=f"加权公允 ${fair:.2f}",
    )
    for trigger_price in (16.0, 12.0):
        fig.add_hline(y=trigger_price, line_color="#dc2626", line_width=1)
    fig.add_trace(go.Scatter(
        x=t_series["period_start"].to_list(),
        y=t_series["value"].to_list(),
        name="T 周收盘", mode="lines+markers",
        line=dict(color="#111827", width=3),
    ))
    fig.update_layout(
        height=460, yaxis_title="$", hovermode="x unified",
        yaxis_range=[10, 32], showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Chart 2: relative performance
# --------------------------------------------------------------------------
st.subheader("相对表现(归一化 = 100)")
if prices.is_empty():
    st.info("等待首次价格采集。")
else:
    lookback = st.radio(
        "窗口", ["26 周", "52 周", "全部"], horizontal=True, key="att_lookback"
    )
    lb = {"26 周": 26, "52 周": 52, "全部": None}[lookback]
    tickers = [t for t, _ in WATCH_TICKERS]
    norm = att_metrics.normalized_performance(prices, tickers, lb)
    fig2 = go.Figure()
    for tk in tickers:
        sub = norm.filter(pl.col("ticker") == tk)
        if sub.is_empty():
            continue
        fig2.add_trace(go.Scatter(
            x=sub["week_start"].to_list(), y=sub["norm"].to_list(),
            name=tk, mode="lines",
            line=dict(
                color="#d97706" if tk == "T" else None,
                width=3 if tk == "T" else 1.5,
            ),
        ))
    fig2.update_layout(
        height=380, yaxis_title="归一化表现", hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "解读: **ASTS 上行 = D2D 多寡头格局 = 对 T 反而是缓冲**;"
        "**SPCX = 威胁情绪 proxy**(IPO 2026-06-12,历史短,自其首周对齐)。"
    )

# --------------------------------------------------------------------------
# Quarterly fundamentals editor
# --------------------------------------------------------------------------
st.subheader("季度基本面(手动录入 → 规则自动评估)")
st.caption(
    "每季财报后录入(下次 **Q2'26 约 7/22**,FCF 指引 $4.0-4.5B)。"
    "churn 阶梯 0.83→0.95→1.10 与融合率/FCF 规则在保存后立即重评。"
)

fund_doc = load_att_json("fundamentals")
quarters = fund_doc.get("quarters", [])
fund_rows = [
    {
        "季度": q.get("quarter", ""),
        "churn %": q.get("churn_pct"),
        "融合率 %": q.get("convergence_pct"),
        "FCF $B": q.get("fcf_billions"),
        "Capex $B": q.get("capex_billions"),
        "净债务/EBITDA": q.get("net_debt_ebitda"),
        "备注": q.get("notes", ""),
    }
    for q in quarters
]
edited_fund = st.data_editor(
    fund_rows or [{
        "季度": "", "churn %": None, "融合率 %": None, "FCF $B": None,
        "Capex $B": None, "净债务/EBITDA": None, "备注": "",
    }],
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "季度": st.column_config.TextColumn(help="格式 2026Q2", width="small"),
        "churn %": st.column_config.NumberColumn(format="%.2f"),
        "融合率 %": st.column_config.NumberColumn(format="%.1f"),
        "FCF $B": st.column_config.NumberColumn(format="%.1f"),
        "Capex $B": st.column_config.NumberColumn(format="%.1f"),
        "净债务/EBITDA": st.column_config.NumberColumn(format="%.2f"),
        "备注": st.column_config.TextColumn(width="large"),
    },
    key="att_fund_editor",
)

if st.button("💾 保存季度数据并重评规则", key="save_fund"):
    new_quarters = []
    for row in edited_fund:
        label = str(row.get("季度", "")).strip()
        if not label:
            continue
        new_quarters.append({
            "quarter": label,
            "churn_pct": row.get("churn %"),
            "convergence_pct": row.get("融合率 %"),
            "fcf_billions": row.get("FCF $B"),
            "capex_billions": row.get("Capex $B"),
            "net_debt_ebitda": row.get("净债务/EBITDA"),
            "notes": str(row.get("备注", "") or ""),
        })
    errors = att_metrics.validate_fundamentals(new_quarters)
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
        # 立即本地重评(不必等周一 CI)
        from ck_trading.monitoring.att_pipeline import build_att_series_provider
        from ck_trading.monitoring.rules import evaluate_rules

        results = evaluate_rules(ATT_RULES, build_att_series_provider(att_store()))
        _store.save_alerts(results)
        load_att_json.clear()
        load_att_alerts.clear()
        st.toast("已保存并重评规则")
        st.rerun()

# mini trend charts
if quarters:
    c1, c2, c3 = st.columns(3)
    churn_s = att_metrics.fundamentals_series(quarters, "churn_pct")
    conv_s = att_metrics.fundamentals_series(quarters, "convergence_pct")
    fcf_s = att_metrics.fundamentals_series(quarters, "fcf_billions")
    with c1:
        if not churn_s.is_empty():
            f = go.Figure(go.Scatter(
                x=churn_s["period_key"].to_list(), y=churn_s["value"].to_list(),
                mode="lines+markers", line=dict(color="#dc2626"),
            ))
            f.add_hline(y=0.95, line_dash="dot", line_color="#d97706")
            f.add_hline(y=1.10, line_dash="dot", line_color="#dc2626")
            f.update_layout(title="churn %", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c2:
        if not conv_s.is_empty():
            f = go.Figure(go.Scatter(
                x=conv_s["period_key"].to_list(), y=conv_s["value"].to_list(),
                mode="lines+markers", line=dict(color="#2563eb"),
            ))
            f.update_layout(title="融合率 %", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c3:
        if not fcf_s.is_empty():
            f = go.Figure(go.Scatter(
                x=fcf_s["period_key"].to_list(), y=fcf_s["value"].to_list(),
                mode="lines+markers", line=dict(color="#16a34a"),
            ))
            f.add_hline(y=4.0, line_dash="dot", line_color="#dc2626")
            f.update_layout(title="季度 FCF $B", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)

# --------------------------------------------------------------------------
# Scenario probability editor
# --------------------------------------------------------------------------
st.subheader("估值情景与主观概率")
st.caption("概率与隐含价可编辑;加权公允价与上方图表实时联动。保存需概率和 = 100%。")

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
    scen_rows,
    use_container_width=True,
    hide_index=True,
    disabled=["情景", "_id"],
    column_config={
        "_id": None,  # hide
        "概率 %": st.column_config.NumberColumn(format="%.1f", min_value=0,
                                              max_value=100),
        "隐含价 $": st.column_config.NumberColumn(format="%.1f"),
        "带下沿 $": st.column_config.NumberColumn(format="%.1f"),
        "带上沿 $": st.column_config.NumberColumn(format="%.1f"),
    },
    key="att_scen_editor",
)

live_scen = [
    {
        "id": row["_id"],
        "label": row["情景"],
        "probability_pct": row["概率 %"],
        "implied_price": row["隐含价 $"],
        "price_low": row["带下沿 $"],
        "price_high": row["带上沿 $"],
    }
    for row in edited_scen
]
live_errors = att_metrics.validate_scenarios(live_scen)
prob_sum = sum(float(r["概率 %"] or 0) for r in edited_scen)
live_fair = att_metrics.weighted_fair_value(live_scen)

sc1, sc2, sc3 = st.columns(3)
sc1.metric("Σ 概率", f"{prob_sum:.1f}%",
           delta="OK" if abs(prob_sum - 100) < 0.01 else "必须 = 100%",
           delta_color="off")
sc2.metric("编辑后加权公允", f"${live_fair:.2f}")
if latest_t_close:
    sc3.metric("现价 vs 编辑后公允", f"{latest_t_close / live_fair - 1:+.1%}")

if live_errors:
    st.warning(" / ".join(live_errors))

if st.button("💾 保存情景", key="save_scen", disabled=bool(live_errors)):
    _store.save_json("scenarios", {
        "version": 1,
        "updated_at": dt.datetime.now(dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenarios": live_scen,
    })
    load_att_json.clear()
    st.toast("情景已保存")
    st.rerun()

# --------------------------------------------------------------------------
# Manual checklist
# --------------------------------------------------------------------------
st.subheader("手动检查清单(报告观测指标 1/2/4 + 辅助项)")
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
    cl_rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Due?": st.column_config.TextColumn(disabled=True, width="small"),
        "项目": st.column_config.TextColumn(disabled=True, width="large"),
        "链接": st.column_config.LinkColumn(disabled=True),
        "周期(天)": st.column_config.NumberColumn(disabled=True, width="small"),
        "上次核查": st.column_config.TextColumn(help="YYYY-MM-DD"),
        "备注": st.column_config.TextColumn(width="large"),
    },
    key="att_checklist_editor",
)
if st.button("💾 保存清单", key="save_att_checklist"):
    for it, row in zip(items, edited_cl):
        raw = str(row.get("上次核查", "")).strip()
        it["last_checked"] = raw or None
        it["notes"] = str(row.get("备注", "") or "")
    _store.save_checklist(items)
    st.toast("已保存到 data/monitoring/att/checklist.json")
    st.rerun()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("论点速查")
    st.markdown(
        "- churn 阶梯: **0.83 → 0.95 → 1.10%**\n"
        "- 融合率基线: **~45%**(连平两季=见顶)\n"
        "- Q2'26 FCF 指引: **$4.0-4.5B**(~7/22)\n"
        "- Starlink 定价解读: $20-30=补充 / **$60+=正面进攻**(T ARPU $85+)\n"
        "- 卫星主承载上限: **<20 人/km²**;物理硬顶 100 人/km²\n"
        "- 卫星射程内美国人口: **2-4%**(叠加定价权受压 ≤10-14%)"
    )
    st.divider()
    st.markdown("**监控标的**")
    st.dataframe(
        [{"ticker": t, "角色": r} for t, r in WATCH_TICKERS],
        use_container_width=True, hide_index=True,
    )
