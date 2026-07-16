"""User-editable configuration for the NFLX framework monitor (4th instance).

Thesis (July 2026): NFLX is mid-transition from growth-stock to value-stock
ownership. Revenue decelerating (17.6% -> 16.2% -> 13.4% -> 11.7% guide) while
the margin/buyback machine strengthens (Q2'26 margin 33.4%, record $4.7B
quarterly buyback, $27.1B authorization). The framework is a pre-committed
BUY LADDER with invalidation rules, not a directional bet:

    | signal              | level                | action                  |
    |---------------------|----------------------|-------------------------|
    | ladder tranche 1    | price <= $70         | buy $10-15K             |
    | ladder tranche 2    | price <= $63         | buy $10-15K             |
    | trend repair        | close > 200dma       | system dip-buy eligible |
    | revenue growth <10% | quarterly            | ladder INVALID, reassess|
    | ads off-track       | quarterly (manual)   | ladder INVALID          |

Scenario seeds (post-Q2'26-earnings probabilities): bull $108/25%,
base $79/55%, bear $54/20% -> weighted fair ~$80.05.

NFLX fiscal quarters == calendar quarters (no label translation needed).
Rule thresholds are static copies — editing scenarios.json never rewrites
alert semantics (same convention as the other monitors).
"""

from __future__ import annotations

from ck_trading.monitoring.rules import AlertRule

WATCH_TICKERS: tuple[tuple[str, str], ...] = (
    ("NFLX", "subject"),
    ("DIS", "competitor"),
    ("SPY", "benchmark"),
    ("QQQ", "growth_benchmark"),
)

NFLX_DEDUPE_KEYS: dict[str, list[str]] = {"prices": ["ticker", "date"]}

DEFAULT_SCENARIOS: tuple[dict, ...] = (
    {"id": "bull", "label": "牛: 广告接棒, FY27 指引重燃增长叙事",
     "probability_pct": 25.0, "implied_price": 108.0,
     "price_low": 100.0, "price_high": 116.0},
    {"id": "base", "label": "基准: 成熟磨底, 利润率+回购扛住减速",
     "probability_pct": 55.0, "implied_price": 79.0,
     "price_low": 72.0, "price_high": 85.0},
    {"id": "bear", "label": "熊: 增长失守个位数 + 广告不及预期",
     "probability_pct": 20.0, "implied_price": 54.0,
     "price_low": 48.0, "price_high": 60.0},
)
# weighted fair = .25*108 + .55*79 + .20*54 = 81.25

NFLX_RULES: tuple[AlertRule, ...] = (
    # ---- 买点 (buy ladder + trend repair) --------------------------------
    AlertRule(
        rule_id="nflx_buy_ladder_t1",
        description="周收盘进入首批买入区间 (<= $70)",
        metric_key="nflx.price_weekly_close",
        dimensions={"ticker": "NFLX"},
        comparator="lt_consecutive",
        threshold=70.0,
        inclusive=True,
        action_label="买点T1: $68-70 区间 — 按预案执行首批 $10-15K",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nflx_buy_ladder_t2",
        description="周收盘进入二批买入区间 (<= $63)",
        metric_key="nflx.price_weekly_close",
        dimensions={"ticker": "NFLX"},
        comparator="lt_consecutive",
        threshold=63.0,
        inclusive=True,
        action_label="买点T2: $60-63 区间 — 二批 $10-15K",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nflx_buy_trend_repair",
        description="收盘重上 200 日线 — 系统 dip 门控重开",
        metric_key="nflx.pct_vs_200dma",
        dimensions={"ticker": "NFLX"},
        comparator="gt_consecutive",
        threshold=0.0,
        action_label="趋势修复: 重上200日线 — dip 信号恢复可用",
        severity="advisory",
    ),
    # ---- 卖点/阶梯失效 (invalidation) ------------------------------------
    AlertRule(
        rule_id="nflx_sell_rev_growth_below_10",
        description="季度营收增速跌破 10% — 增长失守个位数",
        metric_key="nflx.fundamental",
        dimensions={"field": "revenue_growth_pct"},
        comparator="lt_consecutive",
        threshold=10.0,
        max_gap_days=120,
        action_label="失效: 增速进入个位数 — 撤买单, 重审情景概率",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nflx_sell_guide_below_10",
        description="下季指引增速低于 10%",
        metric_key="nflx.fundamental",
        dimensions={"field": "next_q_guide_growth_pct"},
        comparator="lt_consecutive",
        threshold=10.0,
        max_gap_days=120,
        action_label="失效预警: 指引进入个位数 — 阶梯暂停",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nflx_sell_ads_off_track",
        description="广告收入偏离 $3B 路径 (季度人工判定, 1=on track 0=off)",
        metric_key="nflx.fundamental",
        dimensions={"field": "ads_on_track"},
        comparator="lt_consecutive",
        threshold=1.0,
        max_gap_days=120,
        action_label="失效: 广告偏轨 — 牛市情景概率下调, 撤买单",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nflx_sell_margin_below_28",
        description="季度经营利润率跌破 28% — 利润机器失灵",
        metric_key="nflx.fundamental",
        dimensions={"field": "op_margin_pct"},
        comparator="lt_consecutive",
        threshold=28.0,
        max_gap_days=120,
        action_label="失效: 利润率失守 — 基准情景的地板假设破坏",
        severity="trigger",
    ),
)

# ---------------------------------------------------------------------------
# Quarterly fundamentals prefill — from the Q2'26 shareholder letter + xlsx
# (data/Earning/FINAL-Q2-26-Shareholder-Letter.pdf, Q2-26-Website-Financials.xlsx)
# ---------------------------------------------------------------------------
DEFAULT_FUNDAMENTALS: tuple[dict, ...] = (
    {"quarter": "2025Q2", "revenue_growth_pct": 15.9,
     "next_q_guide_growth_pct": None, "op_margin_pct": 34.1,
     "ads_on_track": 1.0, "fcf_billions": 2.267, "buyback_billions": None,
     "notes": "历史基线"},
    {"quarter": "2025Q3", "revenue_growth_pct": 17.2,
     "next_q_guide_growth_pct": None, "op_margin_pct": 28.2,
     "ads_on_track": 1.0, "fcf_billions": 2.660, "buyback_billions": None,
     "notes": "历史基线"},
    {"quarter": "2025Q4", "revenue_growth_pct": 17.6,
     "next_q_guide_growth_pct": None, "op_margin_pct": 24.5,
     "ads_on_track": 1.0, "fcf_billions": 1.872, "buyback_billions": None,
     "notes": "增速顶点"},
    {"quarter": "2026Q1", "revenue_growth_pct": 16.2,
     "next_q_guide_growth_pct": 13.5, "op_margin_pct": 32.3,
     "ads_on_track": 1.0, "fcf_billions": 5.094, "buyback_billions": None,
     "notes": "净利含 WBD $2.8B 分手费(一次性)"},
    {"quarter": "2026Q2", "revenue_growth_pct": 13.4,
     "next_q_guide_growth_pct": 11.7, "op_margin_pct": 33.4,
     "ads_on_track": 1.0, "fcf_billions": 1.525, "buyback_billions": 4.7,
     "notes": "2026-07-16 财报: Q3指引11.7%首破12%; 全年收窄$51.0-51.4B未上调; "
              "广告on track ~$3B; 史上最大单季回购$4.7B, 剩余授权$27.1B; "
              "FCF含WBD费税务拖累(一次性); 观看时长披露降为年度"},
)

DEFAULT_CHECKLIST: tuple[dict, ...] = (
    {"id": "ads_upfront", "label": "US 广告 upfront 承诺落地(财报称'数周内'敲定)— 金额与 CPM 方向",
     "url": "https://about.netflix.com/en/newsroom", "cadence_days": 21},
    {"id": "q3_earnings_entry", "label": "Q3 财报录入(~10月中): 增速/Q4指引/利润率/广告on-track/回购",
     "url": "https://ir.netflix.net/", "cadence_days": 92},
    {"id": "fy27_guide", "label": "FY27 指引(1月, 头号催化剂): 广告翻倍路径 + 增速是否守住双位数",
     "url": "https://ir.netflix.net/", "cadence_days": 180},
    {"id": "pricing_churn", "label": "提价与 churn 动态(Antenna 等第三方数据)— 定价权压力测试",
     "url": "https://www.antenna.live/", "cadence_days": 62},
    {"id": "live_sports", "label": "直播/体育进展: NFL 扩约执行、MLB、拳击 — 广告库存的供给侧",
     "url": "https://about.netflix.com/en/newsroom", "cadence_days": 62},
    {"id": "competitor_moves", "label": "Paramount-WBD 合并体(HBO Max+Paramount+)竞争动作 — 价格战风险",
     "url": "https://www.wbd.com/news", "cadence_days": 62},
)
