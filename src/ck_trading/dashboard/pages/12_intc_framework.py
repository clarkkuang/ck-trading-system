"""INTC framework page — foundry-turnaround / binary-option monitor.

The framework (built July 2026 around the Q2'26 earnings): Intel is a BINARY
turnaround option — does Intel Foundry become a profitable third-party
advanced-node business? Valuation is asset-based (P/B + sum-of-parts, NOT
forward P/E — GAAP is distorted by the CHIPS escrow mark-to-market). The
technical layer is an absolute-price value ladder (T1 $72 / T2 $55 / T3 $40),
NOT a dip-in-uptrend buy (the parabolic run would break that). Initial
verdict at ~$100: SELL-WATCH / trim-into-strength — spot is already above
the bull SOTP (~$92) at a record ~4.6x P/B. See docs/intel_analysis_report.md.
"""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ck_trading.dashboard.data_cache import (
    load_intc_alerts,
    load_intc_json,
    load_intc_monitoring,
)
from ck_trading.monitoring import intc_metrics
from ck_trading.monitoring.intc_config import (
    DEFAULT_SCENARIOS,
    INTC_RULES,
    WATCH_TICKERS,
)
from ck_trading.monitoring.intc_pipeline import (
    build_intc_series_provider,
    intc_store,
    run_weekly_update,
)
from ck_trading.monitoring.quarters import weighted_fair_value

LADDER = [("T1", 72.0), ("T2", 55.0), ("T3", 40.0)]
PB_BUY, PB_SELL = 2.0, 5.0

SCENARIO_COLORS = {
    "bull": "#16a34a", "base": "#2563eb", "bear": "#dc2626", "tail": "#7c2d12",
}

st.set_page_config(page_title="INTC Framework", page_icon="🔷", layout="wide")
st.title("INTC 投资框架监控 — 代工二元期权")
st.caption(
    "**二元反转期权(代工能否盈利?)+ 资产基础估值(P/B+分部, 非前瞻P/E)**。"
    "技术层=绝对价位阶梯 T1 ≤$72 / T2 ≤$55 / T3 ≤$40(有意无视趋势门)。"
    "估值层: P/B≤2 买 / ≥5 减(现~4.6x)。论点层: 14A外部firm客户/18A良率/代工季亏/"
    "DCAI增速/AMD份额。**初始判定: 卖出观察区 — 现价~$100已在牛市SOTP($92)之上, "
    "加权公允~$83, 期权溢价已全额支付**。"
)

_store = intc_store()

# --------------------------------------------------------------------------
# Refresh row
# --------------------------------------------------------------------------
refresh_col, status_col = st.columns([1, 4])
with refresh_col:
    run_clicked = st.button(
        "🔄 Run collection now", type="primary", use_container_width=True,
        help="拉最新日线并重评 14 条规则(约 10 秒)",
    )
with status_col:
    if st.session_state.get("intc_run_summary"):
        st.text(st.session_state["intc_run_summary"])

if run_clicked:
    with st.spinner("Collecting prices + evaluating rules…"):
        result = run_weekly_update()
    st.session_state["intc_run_summary"] = result.summary()
    load_intc_monitoring.clear()
    load_intc_alerts.clear()
    load_intc_json.clear()
    st.rerun()

alerts = load_intc_alerts()
prices = load_intc_monitoring("prices")
scenarios_doc = load_intc_json("scenarios")
scenarios = scenarios_doc.get("scenarios") or [dict(s) for s in DEFAULT_SCENARIOS]
fund_doc = load_intc_json("fundamentals")
quarters = fund_doc.get("quarters", [])

# ---- derived header values -------------------------------------------------
indicators = (
    intc_metrics.daily_indicators(prices, "INTC")
    if not prices.is_empty() else pl.DataFrame()
)
latest_ind = (
    indicators.tail(1).row(0, named=True) if not indicators.is_empty() else None
)
fair = weighted_fair_value(scenarios)
pb_now = intc_metrics.latest_pb_ratio(prices, "INTC", quarters) if quarters else None

# Row 1
m1, m2, m3, m4 = st.columns(4)
if latest_ind:
    px = latest_ind["close"]
    m1.metric("INTC 最新收盘", f"${px:.2f}",
              delta=str(latest_ind["date"]), delta_color="off")
    m3.metric("现价 vs 公允", f"{px / fair - 1:+.1%}",
              delta="高于公允=期权溢价已付" if px > fair else "低于公允=安全边际",
              delta_color="off")
    nxt = next((f"{tag} ≤${lvl:.0f}(距{lvl / px - 1:+.0%})"
                for tag, lvl in LADDER if px > lvl), "全部触发")
    m4.metric("阶梯状态", "未触发" if px > LADDER[0][1] else "已进入阶梯",
              delta=f"下一档 {nxt}", delta_color="off")
else:
    m1.metric("INTC 最新收盘", "—")
    m3.metric("现价 vs 公允", "—")
    m4.metric("阶梯状态", "—")
m2.metric("概率加权公允价", f"${fair:.2f}",
          delta="随情景编辑实时重算", delta_color="off")

# Row 2: valuation + technical
v1, v2, v3 = st.columns(3)
if pb_now:
    pb_val = pb_now[0]
    zone = ("🔴 减仓区(≥5)" if pb_val >= PB_SELL
            else "🟢 便宜区(≤2)" if pb_val <= PB_BUY
            else "⚪ 中性(2-5)")
    v1.metric("P/B(估值层核心)", f"{pb_val:.2f}x",
              delta=f"{zone} · BVPS ${pb_now[1]:.2f}", delta_color="off")
else:
    v1.metric("P/B", "—", delta="需录入 book_value_per_share", delta_color="off")
if latest_ind and latest_ind.get("pct_vs_200dma") is not None:
    v2.metric("vs 200日线", f"{latest_ind['pct_vs_200dma']:+.1%}",
              delta="抛物线上涨中, 远高于200日线(趋势修复短期不触发)"
              if latest_ind["pct_vs_200dma"] > 0.2 else "", delta_color="off")
    v3.metric("距60日高回撤",
              f"{latest_ind['dip_pct']:.1%}"
              if latest_ind.get("dip_pct") is not None else "—",
              delta="回撤发生在200日线上方 — 用价位阶梯而非趋势门",
              delta_color="off")

if alerts.get("updated_at"):
    st.caption(f"规则评估: **{alerts['updated_at']}**")
if prices.is_empty():
    st.warning(
        "还没有价格数据 — 先本地跑 `scripts/intc_monitor_update.py --backfill`"
        "(需要 ~200 交易日历史算 200 日线),或点上方按钮。"
    )

# --------------------------------------------------------------------------
# Verdict banner
# --------------------------------------------------------------------------
rule_states = {r["rule_id"]: r for r in alerts.get("rules", [])}
label_of = {r.rule_id: r.action_label for r in INTC_RULES}
sev_of = {r.rule_id: r.severity for r in INTC_RULES}


def _active(prefix: str, trigger_only: bool) -> list:
    out = []
    for rid, s in rule_states.items():
        if not rid.startswith(prefix) or s.get("status") != "triggered":
            continue
        if trigger_only and sev_of.get(rid) != "trigger":
            continue
        out.append((rid, s))
    return out


buy_active = _active("intc_buy_", trigger_only=True)
sell_active = _active("intc_sell_", trigger_only=True)
insufficient = sum(
    1 for s in rule_states.values() if s.get("status") == "insufficient_data"
)
B, S = len(buy_active), len(sell_active)
if B > 0 and S == 0:
    st.success(f"🟢 **该买 — {B} 个 trigger 级买点活跃**: "
               + " · ".join(label_of.get(rid, rid) for rid, _ in buy_active))
elif S > 0 and B == 0:
    st.error(f"🔴 **该卖/减 — {S} 个 trigger 级卖点活跃**: "
             + " · ".join(label_of.get(rid, rid) for rid, _ in sell_active))
elif B > 0 and S > 0:
    st.warning(f"🟠 **信号冲突({B} 买 / {S} 卖)— 论点层优先于技术层**")
else:
    st.info("⚪ **持有/观望 — 无 trigger 级信号**(初始判定: 卖出观察区, 不在现价加仓)")
if insufficient:
    st.caption(f"{insufficient}/14 规则数据不足(录入季度数据后自动评估; "
               "18A良率官方未披露, 靠清单人工跟踪)")

# --------------------------------------------------------------------------
# Buy / sell card groups
# --------------------------------------------------------------------------
def _fmt_value(rule, value):
    if value is None:
        return "—"
    mk = rule.metric_key
    field = rule.dimensions.get("field", "")
    if mk == "intc.price_weekly_close":
        return f"${value:.2f}"
    if mk == "intc.pct_vs_200dma":
        return f"{value:+.1%}"
    if mk == "intc.pb_ratio":
        return f"{value:.2f}x"
    if field == "foundry_op_loss_billions":
        return f"${value:.2f}B"
    if field in ("dcai_yoy_growth_pct", "amd_server_rev_share_pct",
                 "yield_18a_profitable_pct"):
        return f"{value:.1f}%"
    if field == "adjusted_fcf_billions":
        return f"${value:.1f}B"
    if field == "foundry_14a_firm_customers":
        return f"{value:.0f} 家"
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

buy_rules = [r for r in INTC_RULES if r.rule_id.startswith("intc_buy_")]
sell_rules = [r for r in INTC_RULES if r.rule_id.startswith("intc_sell_")]


def _render(rules, badges, prefix):
    cols = st.columns(len(rules))
    for col, rule in zip(cols, rules):
        state = rule_states.get(rule.rule_id, {})
        status = state.get("status", "insufficient_data")
        with col:
            sev_tag = "" if rule.severity == "trigger" else " ·adv"
            st.metric(
                label=rule.rule_id.removeprefix(prefix) + sev_tag,
                value=_fmt_value(rule, state.get("metric_value")),
                delta=f"streak {state.get('streak', 0)}/{rule.consecutive_periods}",
                delta_color="off",
            )
            badge, kind = badges.get(status, ("?", "warning"))
            getattr(st, kind)(badge)


st.subheader("买点 Buy signals")
_render(buy_rules, BUY_BADGE, "intc_buy_")
st.subheader("卖点 / 失效 Sell signals")
_render(sell_rules, SELL_BADGE, "intc_sell_")

with st.expander("规则定义与告警历史"):
    st.dataframe(
        [{"rule": r.rule_id, "条件": r.description, "动作": r.action_label,
          "级别": r.severity,
          "状态": rule_states.get(r.rule_id, {}).get("status", "—")}
         for r in INTC_RULES],
        use_container_width=True, hide_index=True,
    )
    hist_rows = []
    for rid, state in rule_states.items():
        for h in state.get("history", []):
            hist_rows.append({"rule": rid, "period": h.get("period_key"),
                              "value": h.get("value"), "status": h.get("status")})
    if hist_rows:
        st.dataframe(hist_rows, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Chart 1: INTC weekly close vs scenario bands + value ladder
# --------------------------------------------------------------------------
st.subheader("INTC 周收盘 vs 情景带与价值阶梯")
if prices.is_empty():
    st.info("等待价格数据。")
else:
    closes_w = intc_metrics.weekly_close_series(prices, "INTC")
    cutoff = closes_w["period_start"].max() - dt.timedelta(weeks=156)
    closes_w = closes_w.filter(pl.col("period_start") >= cutoff)
    sma_w = intc_metrics.weekly_sample(indicators, "sma_200").filter(
        pl.col("period_start") >= cutoff
    )

    fig = go.Figure()
    for s in scenarios:
        color = SCENARIO_COLORS.get(s["id"], "#6b7280")
        fig.add_hrect(y0=s["price_low"], y1=s["price_high"],
                      fillcolor=color, opacity=0.07, line_width=0)
        fig.add_hline(
            y=s["implied_price"], line_dash="dot", line_color=color,
            annotation_text=f'${s["implied_price"]:.0f} ({s["probability_pct"]:.0f}%)',
            annotation_font_size=10,
        )
    fig.add_hline(y=fair, line_dash="dash", line_color="#111827", line_width=2,
                  annotation_text=f"加权公允 ${fair:.2f}")
    for tag, lvl in LADDER:
        fig.add_hline(y=lvl, line_color="#15803d", line_width=1.5,
                      annotation_text=f"买{tag} ${lvl:.0f}", annotation_font_size=10)
    fig.add_trace(go.Scatter(
        x=sma_w["period_start"].to_list(), y=sma_w["value"].to_list(),
        name="SMA200", line=dict(color="#dc2626", width=1, dash="dash")))
    fig.add_trace(go.Scatter(
        x=closes_w["period_start"].to_list(), y=closes_w["value"].to_list(),
        name="INTC 周收盘", mode="lines", line=dict(color="#111827", width=3)))
    fig.update_layout(height=480, yaxis_title="$", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Chart 2: P/B history with buy/sell lines
# --------------------------------------------------------------------------
if quarters:
    pb_series = intc_metrics.pb_ratio_series(prices, "INTC", quarters)
    if not pb_series.is_empty():
        st.subheader("P/B(市净率)历史 — 估值层")
        f2 = go.Figure(go.Scatter(
            x=pb_series["period_start"].to_list(),
            y=pb_series["value"].to_list(),
            mode="lines", line=dict(color="#7c3aed", width=2)))
        f2.add_hline(y=PB_SELL, line_dash="dot", line_color="#dc2626",
                     annotation_text="减仓线 5.0x")
        f2.add_hline(y=PB_BUY, line_dash="dot", line_color="#16a34a",
                     annotation_text="便宜线 2.0x")
        f2.update_layout(height=300, yaxis_title="P/B (x)",
                         margin=dict(t=30, b=20), hovermode="x unified")
        st.plotly_chart(f2, use_container_width=True)
        st.caption("注: P/B 序列自 BVPS 首次录入(2026Q2)起; 现~4.5x 逼近减仓线, "
                   "为 Intel 历史最高(2020垄断期~2.8x, 2025困境期~0.8x)。")

# --------------------------------------------------------------------------
# Relative performance
# --------------------------------------------------------------------------
if not prices.is_empty():
    st.subheader("相对表现(归一化 = 100)")
    lb = {"26 周": 26, "52 周": 52, "全部": None}[
        st.radio("窗口", ["26 周", "52 周", "全部"], horizontal=True,
                 index=1, key="intc_lookback")]
    tickers = [t for t, _ in WATCH_TICKERS]
    norm = intc_metrics.normalized_performance(prices, tickers, lb)
    f3 = go.Figure()
    for tk in tickers:
        sub = norm.filter(pl.col("ticker") == tk)
        if sub.is_empty():
            continue
        f3.add_trace(go.Scatter(
            x=sub["week_start"].to_list(), y=sub["norm"].to_list(), name=tk,
            line=dict(width=3 if tk == "INTC" else 1.5)))
    f3.update_layout(height=340, yaxis_title="归一化", hovermode="x unified",
                     legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(f3, use_container_width=True)
    st.caption("角色: AMD=代工/产品双重对手(DC份额攻防) · SMH=半导体β · "
               "SPY=大盘基准。")

# --------------------------------------------------------------------------
# Quarterly fundamentals editor
# --------------------------------------------------------------------------
st.subheader("季度基本面(手动录入 → 规则自动评估)")
st.caption(
    "每季财报后录入(下次 **Q3 财报 ~2026-10**;Intel 财季=日历季)。**GAAP 全面弃用**"
    "(CHIPS 托管按市值计价噪声),以 Non-GAAP + 分部经营利润 + P/B 为轴。"
    "代工亏损/DCAI增速/FCF/EPS 可为负;**18A 良率官方从未披露, 留空即可**"
    "(录入 press 估计会误触发卖点)。保存后规则立即重评。"
)

FIELD_COLS = [
    ("quarter", "季度", "text"),
    ("foundry_external_rev_millions", "代工外部营收 $M", "num"),
    ("foundry_op_loss_billions", "代工季亏 $B", "num"),
    ("foundry_14a_firm_customers", "14A firm客户", "num"),
    ("yield_18a_profitable_pct", "18A盈利良率 %", "num"),
    ("dcai_revenue_billions", "DCAI营收 $B", "num"),
    ("dcai_yoy_growth_pct", "DCAI同比 %", "num"),
    ("amd_server_rev_share_pct", "AMD服务器份额 %", "num"),
    ("adjusted_fcf_billions", "调整后FCF $B", "num"),
    ("non_gaap_eps", "Non-GAAP EPS", "num"),
    ("book_value_per_share", "BVPS(算P/B)", "num"),
    ("notes", "备注", "text"),
]
fund_rows = [{lbl: q.get(fld) for fld, lbl, _ in FIELD_COLS} for q in quarters]
edited_fund = st.data_editor(
    fund_rows or [{lbl: None for _, lbl, _ in FIELD_COLS}],
    num_rows="dynamic", use_container_width=True, hide_index=True,
    key="intc_fund_editor",
)

if st.button("💾 保存季度数据并重评规则", key="save_intc_fund"):
    new_quarters = []
    lbl2fld = {lbl: fld for fld, lbl, _ in FIELD_COLS}
    for row in edited_fund:
        label = str(row.get("季度", "") or "").strip()
        if not label:
            continue
        q = {}
        for lbl, val in row.items():
            fld = lbl2fld.get(lbl)
            if fld == "notes":
                q[fld] = str(val or "")
            elif fld == "quarter":
                q[fld] = label
            elif fld:
                q[fld] = val if val == val else None  # NaN -> None
        new_quarters.append(q)
    errors = intc_metrics.validate_fundamentals(new_quarters)
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
            INTC_RULES, build_intc_series_provider(intc_store()))
        _store.save_alerts(results)
        load_intc_json.clear()
        load_intc_alerts.clear()
        st.toast("已保存并重评规则")
        st.rerun()

# mini trend charts
if quarters:
    from ck_trading.monitoring.quarters import fundamentals_series
    c1, c2, c3 = st.columns(3)
    fl = fundamentals_series(quarters, "foundry_op_loss_billions")
    dc = fundamentals_series(quarters, "dcai_yoy_growth_pct")
    am = fundamentals_series(quarters, "amd_server_rev_share_pct")
    with c1:
        if not fl.is_empty():
            f = go.Figure(go.Scatter(
                x=fl["period_key"].to_list(), y=fl["value"].to_list(),
                mode="lines+markers", line=dict(color="#dc2626")))
            f.update_layout(title="代工季亏 $B(收窄=在轨)", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c2:
        if not dc.is_empty():
            f = go.Figure(go.Scatter(
                x=dc["period_key"].to_list(), y=dc["value"].to_list(),
                mode="lines+markers", line=dict(color="#2563eb")))
            f.update_layout(title="DCAI 同比 %(连2季降=周期顶)", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)
    with c3:
        if not am.is_empty():
            f = go.Figure(go.Bar(
                x=am["period_key"].to_list(), y=am["value"].to_list(),
                marker_color="#f59e0b"))
            f.add_hline(y=50, line_dash="dot", line_color="#dc2626")
            f.update_layout(title="AMD 服务器份额 %(红线=50%)", height=240,
                            margin=dict(t=40, b=20))
            st.plotly_chart(f, use_container_width=True)

# --------------------------------------------------------------------------
# Scenario probability editor
# --------------------------------------------------------------------------
st.subheader("估值情景与主观概率")
st.caption("概率与价格可编辑;加权公允与上方图表实时联动。保存需概率和=100%。")
scen_rows = [{"情景": s["id"], "标签": s["label"], "概率 %": s["probability_pct"],
              "隐含价 $": s["implied_price"], "带下沿 $": s["price_low"],
              "带上沿 $": s["price_high"]} for s in scenarios]
edited_scen = st.data_editor(
    scen_rows, use_container_width=True, hide_index=True, key="intc_scen_editor",
)
live_fair = sum(r["概率 %"] * r["隐含价 $"] / 100 for r in edited_scen)
sc1, sc2, sc3 = st.columns(3)
sc1.metric("Σ 概率", f"{sum(r['概率 %'] for r in edited_scen):.1f}%")
sc2.metric("编辑后加权公允", f"${live_fair:.2f}")
if latest_ind:
    sc3.metric("现价 vs 编辑后公允",
               f"{latest_ind['close'] / live_fair - 1:+.1%}")
if st.button("💾 保存情景", key="save_intc_scen"):
    total = sum(r["概率 %"] for r in edited_scen)
    if abs(total - 100) > 0.1:
        st.error(f"概率和为 {total:.1f}%,需=100%")
    else:
        _store.save_json("scenarios", {
            "version": 1,
            "updated_at": dt.datetime.now(dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scenarios": [
                {"id": r["情景"], "label": r["标签"],
                 "probability_pct": r["概率 %"], "implied_price": r["隐含价 $"],
                 "price_low": r["带下沿 $"], "price_high": r["带上沿 $"]}
                for r in edited_scen],
        })
        load_intc_json.clear()
        st.toast("已保存情景")
        st.rerun()

# --------------------------------------------------------------------------
# Manual checklist
# --------------------------------------------------------------------------
st.subheader("手动检查清单(不可自动化的观测项)")
items = _store.load_checklist()
cl_rows = [{"项目": i["label"], "链接": i.get("url", ""),
            "周期(天)": i.get("cadence_days"),
            "上次核查": i.get("last_checked") or "", "备注": i.get("notes", "")}
           for i in items]
edited_cl = st.data_editor(
    cl_rows, use_container_width=True, hide_index=True,
    column_config={
        "项目": st.column_config.TextColumn(disabled=True, width="large"),
        "链接": st.column_config.LinkColumn(disabled=True),
        "周期(天)": st.column_config.NumberColumn(disabled=True, width="small"),
        "上次核查": st.column_config.TextColumn(help="YYYY-MM-DD"),
    },
    key="intc_checklist_editor",
)
if st.button("💾 保存清单", key="save_intc_checklist"):
    for it, row in zip(items, edited_cl):
        it["last_checked"] = str(row.get("上次核查", "")).strip() or None
        it["notes"] = str(row.get("备注", "") or "")
    _store.save_checklist(items)
    st.toast("已保存到 data/monitoring/intc/checklist.json")
    st.rerun()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("框架速查")
    st.markdown(
        "**决定性变量(单一)**\n"
        "- 代工 14A/18A-P **外部已承诺晶圆量** — 真实第三方业务 vs 补贴型内部成本中心\n\n"
        "**价值阶梯(预承诺, 论点未破才执行)**\n"
        "- T1 ≤**$72** / T2 ≤**$55** / T3 ≤**$40**(有意无视趋势门)\n\n"
        "**估值层(P/B, 非前瞻P/E)**\n"
        "- 买 ≤**2.0** / 减 ≥**5.0**(现 ~4.5x, 历史最高)\n"
        "- 硬地板 $20-25(政府$20.47+账面$22)= 软底非硬底\n\n"
        "**论点层失效条件**\n"
        "- 代工季亏重新走阔 / DCAI增速连2季降 / AMD服务器≥50%\n"
        "- 18A年底<70%良率 / FCF连2季负\n\n"
        "**关键锚点(Q2'26, 7/23)**\n"
        "- 营收$16.1B(+25%,15年最快); Non-GAAP EPS$0.42(超预期~翻倍)\n"
        "- DCAI$6.26B(+59%); 代工亏-$2.09B(连续收窄)\n"
        "- GAAP净亏-$11B=CHIPS托管按市值计价(非现金)\n"
        "- capex上调$20B+; FCF-$8.4B; 股息仍暂停\n\n"
        "**基率诚实**: 国家冠军反转完全重估~1/3-1/4; ~80%半导体项目错过初始进度; "
        "AMD反转靠走fabless(与Intel重资本路线相反)—— **成功非基准情形**\n\n"
        "**与页07 dip系统**: 阶梯有意无视趋势门(价值预承诺 vs 动量), "
        "参见 docs/signal_semantics.md\n\n"
        "*初始判定: 卖出观察区 / 择机减仓, 不在现价加仓*"
    )
    st.divider()
    st.markdown("**监控标的**")
    st.dataframe([{"ticker": t, "角色": r} for t, r in WATCH_TICKERS],
                 use_container_width=True, hide_index=True)
