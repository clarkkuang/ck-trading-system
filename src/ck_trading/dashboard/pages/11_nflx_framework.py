"""NFLX framework page — pre-committed buy-ladder monitor.

The framework (built July 2026 around the Q2'26 earnings): NFLX is
transitioning from growth-stock to value-stock ownership. Rather than a
directional bet, this page tracks a BUY LADDER (T1 <= $70, T2 <= $63)
with quarterly invalidation rules (revenue growth <10%, ads off-track,
margin <28%) and a trend-repair signal (reclaim 200dma).
"""

from __future__ import annotations

import datetime as dt
from datetime import date

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ck_trading.dashboard.data_cache import (
    load_nflx_alerts,
    load_nflx_json,
    load_nflx_monitoring,
)
from ck_trading.monitoring import nflx_metrics
from ck_trading.monitoring.nflx_config import (
    DEFAULT_SCENARIOS,
    NFLX_RULES,
    WATCH_TICKERS,
)
from ck_trading.monitoring.nflx_pipeline import (
    build_nflx_series_provider,
    nflx_store,
    run_weekly_update,
)
from ck_trading.monitoring.quarters import weighted_fair_value

LADDER_T1 = 70.0
LADDER_T2 = 63.0

st.set_page_config(page_title="NFLX Framework", page_icon="🎬", layout="wide")
st.title("NFLX 投资框架监控 — 买入阶梯")
st.caption(
    "**成长→价值换手期的预承诺阶梯**(不看持仓成本)。买点: T1 ≤$70 / T2 ≤$63 "
    "/ 趋势修复(重上200日线)。失效条件(季度录入): 营收增速<10% / 下季指引<10% "
    "/ 广告偏离$3B路径 / 利润率<28% — 任一触发即撤买单重审。"
    "情景: 牛$108(FY27广告接棒) / 基准$79(磨底) / 熊$54(失速)。"
)

SCENARIO_COLORS = {
    "bull": "#16a34a",
    "base": "#2563eb",
    "bear": "#dc2626",
}

_store = nflx_store()

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
    if st.session_state.get("nflx_run_summary"):
        st.text(st.session_state["nflx_run_summary"])

if run_clicked:
    with st.spinner("Collecting prices + evaluating rules…"):
        result = run_weekly_update()
    st.session_state["nflx_run_summary"] = result.summary()
    load_nflx_monitoring.clear()
    load_nflx_alerts.clear()
    load_nflx_json.clear()
    st.rerun()

alerts = load_nflx_alerts()
prices = load_nflx_monitoring("prices")
scenarios_doc = load_nflx_json("scenarios")
scenarios = scenarios_doc.get("scenarios") or [dict(s) for s in DEFAULT_SCENARIOS]
fund_doc = load_nflx_json("fundamentals")
quarters = fund_doc.get("quarters", [])

# ---- derived header values -------------------------------------------------
indicators = (
    nflx_metrics.daily_indicators(prices, "NFLX")
    if not prices.is_empty() else pl.DataFrame()
)
latest_ind = (
    indicators.tail(1).row(0, named=True) if not indicators.is_empty() else None
)
fair = weighted_fair_value(scenarios)

# Row 1: price / fair / distance / next tranche
m1, m2, m3, m4 = st.columns(4)
if latest_ind:
    m1.metric("NFLX 最新收盘", f"${latest_ind['close']:.2f}",
              delta=str(latest_ind["date"]), delta_color="off")
    m3.metric("现价 vs 公允", f"{latest_ind['close'] / fair - 1:+.1%}",
              delta="低于公允=市场更悲观" if latest_ind["close"] < fair else "",
              delta_color="off")
    px = latest_ind["close"]
    if px <= LADDER_T2:
        tranche = "T1+T2 双触发区"
    elif px <= LADDER_T1:
        tranche = f"T1 触发区(距T2 {px / LADDER_T2 - 1:+.1%})"
    else:
        tranche = f"未触发(距T1 {px / LADDER_T1 - 1:+.1%})"
    m4.metric("阶梯状态", tranche, delta=f"T1 ${LADDER_T1:.0f} / T2 ${LADDER_T2:.0f}",
              delta_color="off")
else:
    m1.metric("NFLX 最新收盘", "—")
    m3.metric("现价 vs 公允", "—")
    m4.metric("阶梯状态", "—")
m2.metric("概率加权公允价", f"${fair:.2f}",
          delta="随情景编辑实时重算", delta_color="off")

# Row 2: technical layer readout
if latest_ind and latest_ind.get("pct_vs_200dma") is not None:
    t1, t2, t3 = st.columns(3)
    t1.metric("vs 200日线", f"{latest_ind['pct_vs_200dma']:+.1%}",
              delta="趋势修复(dip门控开)" if latest_ind["pct_vs_200dma"] > 0
              else "趋势外(阶梯以价格为准,不看趋势)",
              delta_color="off")
    t2.metric("200日线", f"${latest_ind['sma_200']:.2f}",
              delta="收复此线=趋势修复信号", delta_color="off")
    t3.metric("距60日高回撤", f"{latest_ind['dip_pct']:.1%}"
              if latest_ind.get("dip_pct") is not None else "—")

if alerts.get("updated_at"):
    st.caption(f"规则评估: **{alerts['updated_at']}**")
if prices.is_empty():
    st.warning(
        "还没有价格数据 — 先本地跑 `scripts/nflx_monitor_update.py --backfill`"
        "(需要 ~200 交易日历史才能算 200 日线),或点上方按钮。"
    )

# --------------------------------------------------------------------------
# Verdict banner
# --------------------------------------------------------------------------
rule_states = {r["rule_id"]: r for r in alerts.get("rules", [])}
buy_active = [
    (rid, s) for rid, s in rule_states.items()
    if rid.startswith("nflx_buy_") and s.get("status") == "triggered"
]
sell_active = [
    (rid, s) for rid, s in rule_states.items()
    if rid.startswith("nflx_sell_") and s.get("status") == "triggered"
]
insufficient = sum(
    1 for s in rule_states.values() if s.get("status") == "insufficient_data"
)
label_of = {r.rule_id: r.action_label for r in NFLX_RULES}

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
    if mk == "nflx.price_weekly_close":
        return f"${value:.2f}"
    if mk == "nflx.pct_vs_200dma":
        return f"{value:+.1%}"
    if field in ("revenue_growth_pct", "next_q_guide_growth_pct"):
        return f"{value:+.1f}%"
    if field == "op_margin_pct":
        return f"{value:.1f}%"
    if field == "ads_on_track":
        return "on track" if value >= 1 else "OFF TRACK"
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

buy_rules = [r for r in NFLX_RULES if r.rule_id.startswith("nflx_buy_")]
sell_rules = [r for r in NFLX_RULES if r.rule_id.startswith("nflx_sell_")]

st.subheader("买点 Buy signals")
cols = st.columns(len(buy_rules))
for col, rule in zip(cols, buy_rules):
    state = rule_states.get(rule.rule_id, {})
    status = state.get("status", "insufficient_data")
    with col:
        st.metric(
            label=rule.rule_id.removeprefix("nflx_buy_"),
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
            label=rule.rule_id.removeprefix("nflx_sell_"),
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
            for r in NFLX_RULES
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
# Chart 1: NFLX weekly close vs scenario bands + buy-ladder lines
# --------------------------------------------------------------------------
st.subheader("NFLX 周收盘 vs 情景带与买入阶梯")
if prices.is_empty():
    st.info("等待价格数据。")
else:
    closes_w = nflx_metrics.weekly_close_series(prices, "NFLX")
    # last ~3 years for readability
    cutoff = closes_w["period_start"].max() - dt.timedelta(weeks=156)
    closes_w = closes_w.filter(pl.col("period_start") >= cutoff)
    sma_w = nflx_metrics.weekly_sample(indicators, "sma_200").filter(
        pl.col("period_start") >= cutoff
    )
    high_w = nflx_metrics.weekly_sample(indicators, "high_60d").filter(
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
    fig.add_hline(y=LADDER_T1, line_color="#16a34a", line_width=2,
                  annotation_text=f"买点T1 ${LADDER_T1:.0f}")
    fig.add_hline(y=LADDER_T2, line_color="#15803d", line_width=2,
                  annotation_text=f"买点T2 ${LADDER_T2:.0f}")
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
        name="NFLX 周收盘", mode="lines",
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
    dip_w = nflx_metrics.weekly_sample(indicators, "dip_pct").filter(
        pl.col("period_start") >= cutoff2
    )
    vs_w = nflx_metrics.weekly_sample(indicators, "pct_vs_200dma").filter(
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
                        key="nflx_lookback")
    lb = {"26 周": 26, "52 周": 52, "全部": None}[lookback]
    tickers = [t for t, _ in WATCH_TICKERS]
    norm = nflx_metrics.normalized_performance(prices, tickers, lb)
    fig3 = go.Figure()
    for tk in tickers:
        sub = norm.filter(pl.col("ticker") == tk)
        if sub.is_empty():
            continue
        fig3.add_trace(go.Scatter(
            x=sub["week_start"].to_list(), y=sub["norm"].to_list(),
            name=tk, mode="lines",
            line=dict(color="#e50914" if tk == "NFLX" else None,
                      width=3 if tk == "NFLX" else 1.5),
        ))
    fig3.update_layout(height=380, yaxis_title="归一化表现",
                       hovermode="x unified",
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "角色: **SPY/QQQ**=市场与成长基准(NFLX跑输=换手仍在进行) · "
        "**DIS**=流媒体同业(共同承压=行业问题, 分化=个股问题)"
    )

# --------------------------------------------------------------------------
# Chart 4: distance to ladder tranches
# --------------------------------------------------------------------------
st.subheader("现价距阶梯档位")
closes_all = nflx_metrics.weekly_close_series(prices, "NFLX")
if closes_all.is_empty():
    st.info("等待价格数据。")
else:
    recent = closes_all.tail(78)  # ~18 months
    f4 = go.Figure(go.Scatter(
        x=recent["period_start"].to_list(),
        y=[v / LADDER_T1 - 1 for v in recent["value"].to_list()],
        mode="lines", name="距T1 %", line=dict(color="#111827"),
    ))
    f4.add_hline(y=0, line_color="#16a34a",
                 annotation_text="T1 触发线 ($70)")
    f4.add_hline(y=LADDER_T2 / LADDER_T1 - 1, line_color="#15803d",
                 annotation_text="T2 触发线 ($63)")
    f4.update_layout(height=320, yaxis_title="现价 / T1 − 1",
                     yaxis_tickformat="+.0%", hovermode="x unified")
    st.plotly_chart(f4, use_container_width=True)

# --------------------------------------------------------------------------
# Quarterly fundamentals editor
# --------------------------------------------------------------------------
st.subheader("季度基本面(手动录入 → 规则自动评估)")
st.caption(
    "每季财报后录入(下次 **Q3 财报 ~2026-10 中,录成 2026Q3**;NFLX 财季=日历季)。"
    "负数合法(增速转负正是信号); 广告on-track 填 1(在轨)或 0(偏离)。保存后规则立即重评。"
)

fund_rows = [
    {
        "季度": q.get("quarter", ""),
        "营收增速 %": q.get("revenue_growth_pct"),
        "下季指引 %": q.get("next_q_guide_growth_pct"),
        "经营利润率 %": q.get("op_margin_pct"),
        "广告on-track": q.get("ads_on_track"),
        "FCF $B": q.get("fcf_billions"),
        "回购 $B": q.get("buyback_billions"),
        "备注": q.get("notes", ""),
    }
    for q in quarters
]
edited_fund = st.data_editor(
    fund_rows or [{
        "季度": "", "营收增速 %": None, "下季指引 %": None,
        "经营利润率 %": None, "广告on-track": None, "FCF $B": None,
        "回购 $B": None, "备注": "",
    }],
    num_rows="dynamic", use_container_width=True, hide_index=True,
    column_config={
        "季度": st.column_config.TextColumn(help="格式 2026Q3", width="small"),
        "营收增速 %": st.column_config.NumberColumn(format="%.1f"),
        "下季指引 %": st.column_config.NumberColumn(format="%.1f"),
        "经营利润率 %": st.column_config.NumberColumn(format="%.1f"),
        "广告on-track": st.column_config.NumberColumn(
            format="%.0f", help="1=在轨, 0=偏离", min_value=0, max_value=1),
        "FCF $B": st.column_config.NumberColumn(format="%.2f"),
        "回购 $B": st.column_config.NumberColumn(format="%.1f"),
        "备注": st.column_config.TextColumn(width="large"),
    },
    key="nflx_fund_editor",
)

if st.button("💾 保存季度数据并重评规则", key="save_nflx_fund"):
    new_quarters = []
    for row in edited_fund:
        label = str(row.get("季度", "")).strip()
        if not label:
            continue
        new_quarters.append({
            "quarter": label,
            "revenue_growth_pct": row.get("营收增速 %"),
            "next_q_guide_growth_pct": row.get("下季指引 %"),
            "op_margin_pct": row.get("经营利润率 %"),
            "ads_on_track": row.get("广告on-track"),
            "fcf_billions": row.get("FCF $B"),
            "buyback_billions": row.get("回购 $B"),
            "notes": str(row.get("备注", "") or ""),
        })
    errors = nflx_metrics.validate_fundamentals(new_quarters)
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
            NFLX_RULES, build_nflx_series_provider(nflx_store())
        )
        _store.save_alerts(results)
        load_nflx_json.clear()
        load_nflx_alerts.clear()
        st.toast("已保存并重评规则")
        st.rerun()

# mini trend charts
if quarters:
    from ck_trading.monitoring.quarters import fundamentals_series

    c1, c2, c3 = st.columns(3)
    rev = fundamentals_series(quarters, "revenue_growth_pct")
    om = fundamentals_series(quarters, "op_margin_pct")
    bb = fundamentals_series(quarters, "buyback_billions")
    with c1:
        if not rev.is_empty():
            f = go.Figure(go.Scatter(
                x=rev["period_key"].to_list(), y=rev["value"].to_list(),
                mode="lines+markers", line=dict(color="#e50914"),
            ))
            f.add_hline(y=10, line_dash="dot", line_color="#dc2626")
            f.update_layout(title="营收增速 %(红线=10%失效线)", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c2:
        if not om.is_empty():
            f = go.Figure(go.Scatter(
                x=om["period_key"].to_list(), y=om["value"].to_list(),
                mode="lines+markers", line=dict(color="#2563eb"),
            ))
            f.add_hline(y=28, line_dash="dot", line_color="#dc2626")
            f.update_layout(title="经营利润率 %(红线=28%)", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c3:
        if not bb.is_empty():
            f = go.Figure(go.Bar(
                x=bb["period_key"].to_list(), y=bb["value"].to_list(),
                marker_color="#16a34a",
            ))
            f.update_layout(title="季度回购 $B", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)

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
    key="nflx_scen_editor",
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
if st.button("💾 保存情景", key="save_nflx_scen", disabled=bool(live_errors)):
    _store.save_json("scenarios", {
        "version": 1,
        "updated_at": dt.datetime.now(dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenarios": live_scen,
    })
    load_nflx_json.clear()
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
    key="nflx_checklist_editor",
)
if st.button("💾 保存清单", key="save_nflx_checklist"):
    for it, row in zip(items, edited_cl):
        raw = str(row.get("上次核查", "")).strip()
        it["last_checked"] = raw or None
        it["notes"] = str(row.get("备注", "") or "")
    _store.save_checklist(items)
    st.toast("已保存到 data/monitoring/nflx/checklist.json")
    st.rerun()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("框架速查")
    st.markdown(
        "**买入阶梯(预承诺)**\n"
        "- T1: **≤$70** 首批 $10-15K\n"
        "- T2: **≤$63** 二批 $10-15K\n"
        "- 趋势修复(重上200日线)= dip 门控重开\n\n"
        "**失效条件(任一触发→撤单重审)**\n"
        "- 营收增速 <**10%** / 下季指引 <**10%**\n"
        "- 广告偏离 **$3B** 路径 / 利润率 <**28%**\n\n"
        "**关键锚点(Q2'26 财报, 7/16)**\n"
        "- 增速轨迹: 17.6→16.2→13.4→**11.7%**(Q3指引)\n"
        "- 利润率 33.4%; 史上最大回购 $4.7B/季, 授权余 $27.1B\n"
        "- 广告 on track ~$3B(翻倍); FY 收窄$51.0-51.4B未上调\n"
        "- 财报日基率: **±9%**(十年中位) — 财报周勿窄带预测\n\n"
        "**催化剂**: Q3财报(~10月) → **FY27指引(1月, 头号)**\n\n"
        "*持仓 725 股 @ $28.74 — 仅作参考,信号不看成本*"
    )
    st.divider()
    st.markdown("**监控标的**")
    st.dataframe(
        [{"ticker": t, "角色": r} for t, r in WATCH_TICKERS],
        use_container_width=True, hide_index=True,
    )
