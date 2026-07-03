"""AI Model Share page — open-source/Chinese-lab threat monitor for Anthropic.

Live view of the monitoring pipeline in ck_trading.monitoring:
rule status cards, dollar-weighted bloc share chart, npm/PyPI download
curves, Claude flagship price timeline, and the L2/L3 manual checklist.
"""

from __future__ import annotations

import datetime as dt
from datetime import date

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ck_trading.dashboard.data_cache import (
    load_monitoring,
    load_monitoring_alerts,
)
from ck_trading.monitoring import metrics
from ck_trading.monitoring.blocs import ORG_BLOC_MAP
from ck_trading.monitoring.rules_config import FLAGSHIP_FAMILIES, RULES
from ck_trading.monitoring.store import MonitoringStore

st.set_page_config(page_title="AI Model Share", page_icon="📡", layout="wide")
st.title("AI Model Share Monitor")
st.caption(
    "开源/中国系模型对 Anthropic 的竞争威胁 — 周频 L1 指标自动采集,"
    "核心是 OpenRouter **美元加权份额**(token 量 × 挂牌价,按阵营分组)。"
    "详见 docs/ai_model_share_monitoring.md。数据引用:Source: OpenRouter "
    "(openrouter.ai/rankings)。"
)

BLOC_COLORS = {
    "chinese": "#dc2626",
    "anthropic": "#d97706",
    "western_closed": "#2563eb",
    "western_open": "#0891b2",
    "unclassified": "#9ca3af",
    "other": "#d1d5db",
}
STATUS_BADGE = {
    "triggered": ("🔴 TRIGGERED", "error"),
    "ok": ("🟢 OK", "success"),
    "insufficient_data": ("🟡 insufficient data", "warning"),
}


# --------------------------------------------------------------------------
# Refresh row
# --------------------------------------------------------------------------
refresh_col, status_col = st.columns([1, 4])
with refresh_col:
    run_clicked = st.button(
        "🔄 Run collection now",
        type="primary",
        use_container_width=True,
        help="采集 OpenRouter/npm/PyPI 最新数据并重算所有规则(约 1-3 分钟)",
    )
with status_col:
    if st.session_state.get("ai_share_run_summary"):
        st.text(st.session_state["ai_share_run_summary"])

if run_clicked:
    from ck_trading.monitoring.pipeline import run_weekly_update

    with st.spinner("Collecting OpenRouter + npm/PyPI data…"):
        result = run_weekly_update()
    st.session_state["ai_share_run_summary"] = result.summary()
    load_monitoring.clear()
    load_monitoring_alerts.clear()
    st.rerun()

alerts = load_monitoring_alerts()
weekly = load_monitoring("weekly_bloc_share")
pricing = load_monitoring("openrouter_pricing_snapshots")
downloads = load_monitoring("pkg_downloads")

freshness_bits = []
if not weekly.is_empty():
    freshness_bits.append(f"bloc share 至 **{weekly['week_start'].max()}**")
if not pricing.is_empty():
    freshness_bits.append(f"定价快照 **{pricing['snapshot_date'].max()}**")
if alerts.get("updated_at"):
    freshness_bits.append(f"规则评估 **{alerts['updated_at']}**")
if freshness_bits:
    st.caption("数据新鲜度: " + " · ".join(freshness_bits))

if weekly.is_empty() and pricing.is_empty() and downloads.is_empty():
    st.warning(
        "还没有监控数据 — 点上方 **Run collection now**,"
        "或等周一的 GitHub Action 自动跑。"
    )
    st.stop()


# --------------------------------------------------------------------------
# Rule status cards
# --------------------------------------------------------------------------
st.subheader("触发规则状态")
rule_states = {r["rule_id"]: r for r in alerts.get("rules", [])}
cols = st.columns(max(len(RULES), 1))
for col, rule in zip(cols, RULES):
    state = rule_states.get(rule.rule_id, {})
    status = state.get("status", "insufficient_data")
    value = state.get("metric_value")
    streak = state.get("streak", 0)
    with col:
        if "share" in rule.metric_key:
            val_str = f"{value:.1%}" if value is not None else "—"
            thr_str = f"{rule.threshold:.0%}"
        elif "price" in rule.metric_key:
            val_str = f"${value:,.2f}/Mtok" if value is not None else "—"
            thr_str = f"-{rule.threshold:.0%}"
        else:
            val_str = f"{value:,.0f}" if value is not None else "—"
            thr_str = f"{rule.consecutive_periods}w decline"
        st.metric(
            label=rule.rule_id,
            value=val_str,
            delta=f"阈值 {thr_str} · streak {streak}/{rule.consecutive_periods}",
            delta_color="off",
        )
        badge, kind = STATUS_BADGE.get(status, ("?", "warning"))
        getattr(st, kind)(f"{badge} → {rule.action_label or rule.severity}")

with st.expander("告警历史 / episode 详情"):
    if rule_states:
        hist_rows = []
        for rid, state in rule_states.items():
            for h in state.get("history", []):
                hist_rows.append({
                    "rule": rid,
                    "period": h.get("period_key"),
                    "value": h.get("value"),
                    "status": h.get("status"),
                })
        if hist_rows:
            st.dataframe(hist_rows, use_container_width=True, hide_index=True)
        ep = [
            {
                "rule": rid,
                "episode_started": s.get("episode_started"),
                "triggered_at": s.get("triggered_at"),
                "resolved_at": s.get("resolved_at"),
            }
            for rid, s in rule_states.items()
            if s.get("episode_started") or s.get("resolved_at")
        ]
        if ep:
            st.markdown("**Episodes**")
            st.dataframe(ep, use_container_width=True, hide_index=True)
    else:
        st.info("还没有规则评估记录。")


# --------------------------------------------------------------------------
# Chart 1: bloc share stacked area
# --------------------------------------------------------------------------
st.subheader("OpenRouter 阵营份额(周度)")
if weekly.is_empty():
    st.info(
        "还没有 rankings 数据 — 需要 OPENROUTER_API_KEY"
        "(免费,openrouter.ai/keys),放 .env 或 GitHub Secret 后下次采集生效。"
    )
else:
    c1, c2 = st.columns([1, 1])
    with c1:
        share_mode = st.radio(
            "口径", ["美元加权", "Token 加权"], horizontal=True, key="share_mode"
        )
    scopes = weekly["scope"].unique().sort().to_list()
    with c2:
        scope_pref = ["programming", "programming:unverified", "all"]
        default_scope = next((s for s in scope_pref if s in scopes), scopes[0])
        scope = st.selectbox(
            "Scope", scopes, index=scopes.index(default_scope), key="share_scope"
        )

    metric_col = "dollar_share" if share_mode == "美元加权" else "token_share"
    sub = weekly.filter(
        (pl.col("scope") == scope) & pl.col(metric_col).is_not_null()
    ).sort("week_start")

    fig = go.Figure()
    for bloc in ["chinese", "anthropic", "western_closed", "western_open",
                 "unclassified", "other"]:
        b = sub.filter(pl.col("bloc") == bloc)
        if b.is_empty():
            continue
        fig.add_trace(go.Scatter(
            x=b["week_start"].to_list(),
            y=(b[metric_col] * 100).to_list(),
            name=bloc, stackgroup="one",
            line=dict(color=BLOC_COLORS.get(bloc, "#6b7280"), width=0.5),
        ))
    if metric_col == "dollar_share":
        fig.add_hline(
            y=35, line_dash="dash", line_color="#dc2626",
            annotation_text="CN 触发线 35%",
        )
    fig.update_layout(
        height=420, yaxis_title=f"{share_mode} 份额 (%)",
        hovermode="x unified", legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # measurement error note
    latest_week = sub["iso_week"].max()
    latest = sub.filter(pl.col("iso_week") == latest_week)
    other_share = latest.filter(pl.col("bloc") == "other")["token_share"].sum()
    uncls_share = (
        latest.filter(pl.col("bloc") == "unclassified")["dollar_share"].sum()
    )
    st.info(
        f"**测量误差**({latest_week}): top-50 截断的 'other' 占 token "
        f"{other_share:.1%};未归类 org 占美元 {uncls_share or 0:.1%}。"
        "另注意: OpenRouter 只是全市场 proxy(不含直连 API/企业合同流量);"
        "美元加权采用冻结的 70/30 prompt:completion 混合价假设。"
    )


# --------------------------------------------------------------------------
# Chart 2: package downloads
# --------------------------------------------------------------------------
st.subheader("开发者采用度: CLI / SDK 周下载")
if downloads.is_empty():
    st.info("还没有下载数据。")
else:
    tab_npm, tab_pypi = st.tabs(["npm (CLI)", "PyPI (SDK)"])
    log_scale = st.toggle("对数坐标", value=True, key="dl_log")
    for tab, registry in ((tab_npm, "npm"), (tab_pypi, "pypi")):
        with tab:
            reg = downloads.filter(pl.col("registry") == registry)
            if reg.is_empty():
                st.info(f"暂无 {registry} 数据。")
                continue
            fig = go.Figure()
            for pkg in reg["package"].unique().sort().to_list():
                p = reg.filter(pl.col("package") == pkg).sort("period_start")
                is_anthropic = "anthropic" in pkg
                fig.add_trace(go.Scatter(
                    x=p["period_start"].to_list(),
                    y=p["downloads"].to_list(),
                    name=pkg, mode="lines+markers",
                    line=dict(
                        color="#d97706" if is_anthropic else None,
                        width=3 if is_anthropic else 1.5,
                    ),
                ))
            fig.update_layout(
                height=380, yaxis_title="周下载量",
                yaxis_type="log" if log_scale else "linear",
                hovermode="x unified", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Chart 3: flagship price timeline
# --------------------------------------------------------------------------
st.subheader("Claude 旗舰挂牌价(completion $/Mtok)")
if pricing.is_empty():
    st.info("还没有定价快照。")
else:
    fig = go.Figure()
    any_series = False
    for family in FLAGSHIP_FAMILIES:
        s = metrics.flagship_price_series(pricing, family)
        if s.is_empty():
            continue
        any_series = True
        fig.add_trace(go.Scatter(
            x=s["period_start"].to_list(), y=s["value"].to_list(),
            name=family.split("/")[-1], mode="lines+markers",
        ))
        # -30% band off the trailing max
        max_val = s["value"].max()
        fig.add_hline(
            y=max_val * 0.7, line_dash="dot", line_color="#dc2626",
            annotation_text=f"{family.split('/')[-1]} -30% 触发线",
        )
    if any_series:
        fig.update_layout(
            height=380, yaxis_title="completion $/Mtok",
            hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("定价快照里没有匹配 FLAGSHIP_FAMILIES 的模型。")


# --------------------------------------------------------------------------
# L2/L3 manual checklist
# --------------------------------------------------------------------------
st.subheader("L2/L3 手动检查清单")
st.caption(
    "无法自动化的验证/确认层指标 — 定期人工核查,超期的行会标 Due。"
    "保存后写入 data/monitoring/checklist.json(commit 后跨机器同步)。"
)

_store = MonitoringStore()
items = _store.load_checklist()
today = date.today()

rows = []
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
    rows.append({
        "Due?": "🔔 DUE" if due else "✓",
        "项目": it["label"],
        "链接": it.get("url", ""),
        "周期(天)": it.get("cadence_days", 31),
        "上次核查": str(last) if last else "",
        "备注": it.get("notes", ""),
    })

edited = st.data_editor(
    rows,
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
    key="checklist_editor",
)

if st.button("💾 保存清单", key="save_checklist"):
    for it, row in zip(items, edited):
        raw = str(row.get("上次核查", "")).strip()
        it["last_checked"] = raw or None
        it["notes"] = str(row.get("备注", "")).strip()
    _store.save_checklist(items)
    st.toast("已保存到 data/monitoring/checklist.json")
    st.rerun()


# --------------------------------------------------------------------------
# Sidebar: rule + bloc reference
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("规则定义")
    st.dataframe(
        [
            {
                "rule": r.rule_id,
                "阈值": r.threshold,
                "连续周": r.consecutive_periods,
                "动作": r.action_label,
            }
            for r in RULES
        ],
        use_container_width=True, hide_index=True,
    )
    with st.expander("阵营分类表(bloc membership)"):
        st.dataframe(
            [{"org": o, "bloc": b} for o, b in sorted(ORG_BLOC_MAP.items())],
            use_container_width=True, hide_index=True,
        )
