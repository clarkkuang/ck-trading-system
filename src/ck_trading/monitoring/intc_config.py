"""User-editable configuration for the Intel framework monitor (5th instance).

Thesis (July 2026): INTC is a BINARY turnaround option — does Intel Foundry
become a profitable third-party advanced-node business? Unlike the growth
(NVDA), value-transfer (NFLX), or defensive (AT&T) instances, Intel's core
variable is discrete, its valuation is asset-based (P/B + sum-of-parts, NOT
forward P/E — GAAP is distorted by the CHIPS escrow mark-to-market and
non-GAAP forward P/E is ~60x at ~$100), and it trades below/around its
200dma after a $19->$142->~$100 near-parabola.

Framework verdict at design (2026-07-23, ~$100): SELL-WATCH / trim-into-
strength. Weighted fair ~$83 vs spot ~$100; price is already above the bull
SOTP (~$92) and at a record 4.6x P/B, so the "cheap option + SOTP floor"
entry thesis has largely played out. New buys are gated to the $72/$55/$40
value ladder and the foundry pivot variable, NOT chased at spot.

See docs/intel_analysis_report.md for the full framework.
Intel fiscal quarters == calendar quarters. Rule thresholds are static
copies — editing scenarios.json never rewrites alert semantics.
"""

from __future__ import annotations

from ck_trading.monitoring.rules import AlertRule

WATCH_TICKERS: tuple[tuple[str, str], ...] = (
    ("INTC", "subject"),
    ("AMD", "competitor"),
    ("SMH", "sector"),
    ("SPY", "benchmark"),
)

INTC_DEDUPE_KEYS: dict[str, list[str]] = {"prices": ["ticker", "date"]}

# weighted fair = .15*170 + .35*95 + .35*57 + .15*30 = 83.20
DEFAULT_SCENARIOS: tuple[dict, ...] = (
    {"id": "bull", "label": "牛/尾部上行: 代工外部营收拐点, 重估为'美国版TSMC'",
     "probability_pct": 15.0, "implied_price": 170.0,
     "price_low": 150.0, "price_high": 190.0},
    {"id": "base", "label": "基准: AI顺风延续但代工未证明, 区间震荡",
     "probability_pct": 35.0, "implied_price": 95.0,
     "price_low": 80.0, "price_high": 110.0},
    {"id": "bear", "label": "熊: AI capex消化+估值压缩, 代工亏损延续",
     "probability_pct": 35.0, "implied_price": 57.0,
     "price_low": 45.0, "price_high": 70.0},
    {"id": "tail", "label": "尾部下行: 代工减值, 战略/政府软底托住",
     "probability_pct": 15.0, "implied_price": 30.0,
     "price_low": 20.0, "price_high": 40.0},
)

INTC_RULES: tuple[AlertRule, ...] = (
    # ---- 买点 · 技术层(NFLX式绝对价位阶梯, 有意无视趋势门) --------------
    AlertRule(
        rule_id="intc_buy_ladder_t1",
        description="周收盘进入首批价值带 (<= $72), 仅论点未破时执行",
        metric_key="intc.price_weekly_close",
        dimensions={"ticker": "INTC"},
        comparator="lt_consecutive", threshold=72.0, inclusive=True,
        action_label="买点T1: <=$72 价值回归带 — 首批(前提: 二元论点未破)",
        severity="trigger",
    ),
    AlertRule(
        rule_id="intc_buy_ladder_t2",
        description="周收盘逼近 base-SOTP (<= $55)",
        metric_key="intc.price_weekly_close",
        dimensions={"ticker": "INTC"},
        comparator="lt_consecutive", threshold=55.0, inclusive=True,
        action_label="买点T2: <=$55 逼近基准分部估值 — 二批",
        severity="trigger",
    ),
    AlertRule(
        rule_id="intc_buy_ladder_t3",
        description="周收盘进入软底带上方 (<= $40)",
        metric_key="intc.price_weekly_close",
        dimensions={"ticker": "INTC"},
        comparator="lt_consecutive", threshold=40.0, inclusive=True,
        action_label="买点T3: <=$40 政府成本/账面软底上方 — 三批/tail保护区",
        severity="trigger",
    ),
    AlertRule(
        rule_id="intc_buy_trend_repair",
        description="重上并连续2周站稳200日线 — dip动量系统恢复可用",
        metric_key="intc.pct_vs_200dma",
        dimensions={"ticker": "INTC"},
        comparator="gt_consecutive", threshold=0.0, consecutive_periods=2,
        action_label="趋势修复: 站稳200日线 — dip信号恢复(为真跌破后夺回预置)",
        severity="advisory",
    ),
    # ---- 买点 · 估值层(P/B) ---------------------------------------------
    AlertRule(
        rule_id="intc_buy_pb_below_2",
        description="P/B <= 2.0 回到历史便宜期权区(现~4.6x远高于此)",
        metric_key="intc.pb_ratio",
        dimensions={"ticker": "INTC"},
        comparator="lt_consecutive", threshold=2.0,
        action_label="估值买点: P/B<=2 安全边际恢复(前提论点未破)",
        severity="advisory",
    ),
    # ---- 买点 · 论点层 ---------------------------------------------------
    AlertRule(
        rule_id="intc_buy_14a_firm_commitment",
        description="14A先进节点外部firm大客户 >=1(现为0, 二元论点最硬向上触发)",
        metric_key="intc.fundamental",
        dimensions={"field": "foundry_14a_firm_customers"},
        comparator="gt_consecutive", threshold=1.0, inclusive=True,
        max_gap_days=120,
        action_label="论点确认: 首个14A外部firm大客户 — 代工真实性验证, 持有/回调加仓",
        severity="trigger",
    ),
    AlertRule(
        rule_id="intc_buy_18a_yield_hit",
        description="18A盈利良率 >=70%(每片盈利, 支撑2027盈亏平衡)",
        metric_key="intc.fundamental",
        dimensions={"field": "yield_18a_profitable_pct"},
        comparator="gt_consecutive", threshold=70.0, inclusive=True,
        max_gap_days=200,
        action_label="论点向上: 18A达盈利良率 — 论点向上",
        severity="advisory",
    ),
    # ---- 卖点 · 技术层 ---------------------------------------------------
    AlertRule(
        rule_id="intc_sell_trend_break",
        description="连续4周收于200日线下方 — 动量破位(advisory, 论点/估值优先)",
        metric_key="intc.pct_vs_200dma",
        dimensions={"ticker": "INTC"},
        comparator="lt_consecutive", threshold=0.0, consecutive_periods=4,
        action_label="趋势破位: 连续4周破200日线 — 降杠杆(论点/估值卖点优先)",
        severity="advisory",
    ),
    # ---- 卖点 · 估值层 ---------------------------------------------------
    AlertRule(
        rule_id="intc_sell_pb_above_5",
        description="P/B >= 5.0(创纪录且高于牛市SOTP~$92)— optionality全额计价",
        metric_key="intc.pb_ratio",
        dimensions={"ticker": "INTC"},
        comparator="gt_consecutive", threshold=5.0,
        action_label="估值卖点: P/B>=5 无历史先例 — 趋势性减仓",
        severity="advisory",
    ),
    # ---- 卖点 · 论点层 ---------------------------------------------------
    AlertRule(
        rule_id="intc_sell_foundry_loss_widen",
        description="代工季度经营亏损QoQ重新走阔(偏离2027盈亏平衡轨道)",
        metric_key="intc.fundamental",
        dimensions={"field": "foundry_op_loss_billions"},
        comparator="non_increasing_streak", threshold=0.0,
        consecutive_periods=1, max_gap_days=120,
        action_label="论点卖点: 代工亏损重新走阔 — 偏离盈亏平衡轨道",
        severity="trigger",
    ),
    AlertRule(
        rule_id="intc_sell_dcai_decel_2q",
        description="DCAI同比增速连续2季下滑(AI capex周期见顶信号)",
        metric_key="intc.fundamental",
        dimensions={"field": "dcai_yoy_growth_pct"},
        comparator="non_increasing_streak", threshold=0.0,
        consecutive_periods=2, max_gap_days=120,
        action_label="论点卖点: DCAI增速连续两季减速 — 周期杠杆现金牛承压",
        severity="trigger",
    ),
    AlertRule(
        rule_id="intc_sell_18a_yield_miss",
        description="至2026Q4 18A盈利良率仍<70%(CFO自设年底红线; 官方数字披露前insufficient)",
        metric_key="intc.fundamental",
        dimensions={"field": "yield_18a_profitable_pct"},
        comparator="lt_consecutive", threshold=70.0, max_gap_days=200,
        action_label="论点卖点: 18A年底仍未达盈利良率 — 盈利时点滑入2027+",
        severity="trigger",
    ),
    AlertRule(
        rule_id="intc_sell_fcf_negative_2q",
        description="调整后自由现金流连续2季为负(capex上调至>$20B后的现金消耗)",
        metric_key="intc.fundamental",
        dimensions={"field": "adjusted_fcf_billions"},
        comparator="lt_consecutive", threshold=0.0, consecutive_periods=2,
        max_gap_days=120,
        action_label="现金预警: FCF连续两季为负 — 股息恢复更远",
        severity="advisory",
    ),
    AlertRule(
        rule_id="intc_sell_amd_server_share_50",
        description="AMD服务器CPU收入份额 >=50%(现46.2%, 过半=x86 DC结构性侵蚀确认)",
        metric_key="intc.fundamental",
        dimensions={"field": "amd_server_rev_share_pct"},
        comparator="gt_consecutive", threshold=50.0, inclusive=True,
        max_gap_days=120,
        action_label="论点卖点: AMD服务器份额过半 — x86数据中心结构性侵蚀确认",
        severity="advisory",
    ),
)

# ---------------------------------------------------------------------------
# Quarterly fundamentals prefill — from the Q2'26 release (2026-07-23)
# yield_18a_profitable_pct intentionally left null: Intel never discloses an
# official yield number; entering the ~55-65 press estimate would falsely
# fire intc_sell_18a_yield_miss before the CFO's year-end-2026 redline.
# Tracked via the intc_18a_yield_redline checklist item instead.
# ---------------------------------------------------------------------------
DEFAULT_FUNDAMENTALS: tuple[dict, ...] = (
    {"quarter": "2026Q1",
     "foundry_external_rev_millions": 174.0, "foundry_op_loss_billions": -2.4,
     "foundry_14a_firm_customers": 0.0, "yield_18a_profitable_pct": None,
     "dcai_revenue_billions": 5.1, "dcai_yoy_growth_pct": 22.0,
     "amd_server_rev_share_pct": 46.2, "adjusted_fcf_billions": None,
     "non_gaap_eps": 0.29, "book_value_per_share": None,
     "notes": "历史基线; 代工外部营收靠推算 ~$174M"},
    {"quarter": "2026Q2",
     "foundry_external_rev_millions": None, "foundry_op_loss_billions": -2.089,
     "foundry_14a_firm_customers": 0.0, "yield_18a_profitable_pct": None,
     "dcai_revenue_billions": 6.262, "dcai_yoy_growth_pct": 59.0,
     "amd_server_rev_share_pct": None, "adjusted_fcf_billions": -8.419,
     "non_gaap_eps": 0.42, "book_value_per_share": 22.18,
     "notes": "2026-07-23财报: 营收$16.1B(+25%,近15年最快); Non-GAAP EPS$0.42"
              "(超预期~翻倍); GAAP净亏-$11B全来自$12.5B CHIPS托管股按市值计价(非现金);"
              " DCAI$6.26B(+59%,创纪录); 代工亏-$2.09B(连续收窄); capex上调至$20B+;"
              " FCF-$8.4B; 股息仍暂停; 外部代工营收仍极小(未单列); 良率官方未披露(~55-65% press估)"},
)

DEFAULT_CHECKLIST: tuple[dict, ...] = (
    {"id": "intc_14a_commitment_window",
     "label": "14A外部firm承诺窗口(最关键向上催化)— 0.9版PDK定于2026-10; 盯2026H2是否出现有量产/预付承诺的先进节点大客户(现0 firm, 2家潜在洽谈; Fortinet仅成熟Intel 4节点不计入)",
     "url": "https://www.intc.com/news-events", "cadence_days": 31},
    {"id": "intc_18a_yield_redline",
     "label": "18A盈利良率对CFO'2026年底'红线 — 是否达70%+(官方从未披露具体数字, press估~55-65%); 滑入2027=论点延迟加深; 有官方Q4数字后录入fundamentals",
     "url": "https://www.intc.com/news-events", "cadence_days": 31},
    {"id": "intc_product_ramp",
     "label": "Panther Lake(客户端,18A)+ Clearwater Forest(服务器,18A)放量/评测 — 验证内部采用这条腿并填充代工产能",
     "url": "https://www.intc.com/news-events", "cadence_days": 31},
    {"id": "intc_govt_stake_warrant",
     "label": "美国政府~10%持股($20.47成本) + $20认股权证(挂钩'代工持股跌破51%'毒丸)状态 — 战略软底与拆分路径约束的变化",
     "url": "https://www.intc.com/news-events", "cadence_days": 92},
    {"id": "intc_dividend_watch",
     "label": "股息恢复观察 — 自2024-09暂停; 恢复需产品份额稳定+代工盈利+去杠杆, 现价无收益支撑",
     "url": "https://www.intc.com/news-events", "cadence_days": 92},
    {"id": "intc_earnings_entry",
     "label": "INTC财报录入(下次~2026-10 Q3'26): 代工外部营收/季亏、18A良率、DCAI/增速、Non-GAAP EPS、FCF、BVPS(算P/B)、AMD服务器份额",
     "url": "https://www.intc.com/news-events", "cadence_days": 92},
    {"id": "intc_amd_dc_head2head",
     "label": "AMD Q2数据中心营收 vs Intel DCAI头对头(AMD~8月初财报)— 核对Q2 +59%反弹是可持续夺回还是一次性AI周期顺风",
     "url": "https://ir.amd.com/", "cadence_days": 92},
)
