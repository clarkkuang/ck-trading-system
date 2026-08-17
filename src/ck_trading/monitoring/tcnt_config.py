"""User-editable configuration for the Tencent (0700.HK) EXIT monitor.

⚠️ This is the 6th instance and it is a DIFFERENT SPECIES from the other five.

The other five answer "buy / hold / sell?". This one does not: the direction
is already decided — the holder is liquidating the entire position over five
years. The only open question is PACING. Consequences for the design:

  * There are NO buy rules. Adding any would contradict the objective.
  * The default action is NOT "do nothing" — it is "sell the scheduled
    tranche". Rules only speed that up or slow it down.
  * The enemy is path regret in both directions: clearing at HK$447 and
    watching it reach HK$667, or holding out for HK$620 and watching it
    revisit the 2022 trough (6.26x P/E — a real observation, not a tail).

WHY THE RAILS ARE MULTIPLES, NOT PRICES
    Over a five-year horizon static price rails decay into uselessness: at 9%
    EPS growth HK$620 is 19x today and 12.3x in 2031. Every rail is therefore
    expressed as a multiple of trailing non-IFRS EPS and recomputed each
    quarter. The band edges themselves come from the trailing 52-week P/E
    range (percentile-based), so they drift with the market instead of being
    hand-picked round numbers.

WHY SOTP IS DELIBERATELY NOT USED AS AN EXIT TARGET
    Sell-side SOTP puts Tencent at HK$652-725 (core + investment portfolio +
    net cash - minorities, 10% holdco discount) versus a HK$447 market price.
    That 46-62% discount has persisted for years and never converged — the
    market applies its own discount to an illiquid investment book with no
    realisation path. Anchoring an exit on SOTP fair value means never
    selling. The rails track what the market actually pays.

TAX IS NOT A BINDING CONSTRAINT HERE
    Cost basis ~HK$400 vs ~HK$447 spot is a ~12% embedded gain, unlike the
    NVDA position's 10x. Nothing in this ladder is shaped by tax.

CALIBRATION NOTE (2026-08-16): band edges were moved once, before the first
tranche, from hand-picked integers (10/12/17x) to percentiles of the trailing
52-week P/E range, after the holder judged 13.7x too cheap for a full
tranche. That recalibration is now FROZEN. Subsequent "let's wait" impulses
are answered by the rule, not by moving the rule.
"""

from __future__ import annotations

from ck_trading.monitoring.rules import AlertRule

WATCH_TICKERS: tuple[tuple[str, str], ...] = (
    ("0700.HK", "subject"),
    ("9988.HK", "peer"),      # Alibaba — the other China mega-cap AI spender
    ("3690.HK", "peer"),      # Meituan
    ("^HSI", "benchmark"),    # Hang Seng
)

TCNT_DEDUPE_KEYS: dict[str, list[str]] = {"prices": ["ticker", "date"]}

# ---------------------------------------------------------------------------
# Exit schedule
# ---------------------------------------------------------------------------
#: total tranches; 20 quarters = 5 years (2026Q4 -> 2031Q3)
EXIT_TRANCHES: int = 20
#: baseline fraction of the ORIGINAL position sold per scheduled tranche
EXIT_BASELINE_PCT: float = 5.0
#: the holder elected on 2026-08-16 to defer the first tranche by one full
#: slot. The gate is a rule, not an open-ended wait: whichever comes first.
FIRST_TRANCHE_GATE = {
    "not_before_quarter": "2027Q1",
    "or_price_at_or_above_pe": 15.4,   # mid-band of the 52w range at design
    "hard_deadline": "2027-03-31",
}
#: deep-value skips are budgeted so "wait" cannot become "never"
SKIP_BUDGET: int = 4

# ---------------------------------------------------------------------------
# Valuation bands — percentiles of the TRAILING 52-WEEK P/E range.
# Recomputed every quarter by tcnt_metrics.exit_band(); the numbers below are
# the design-date snapshot (52w range 12.6x-20.9x, spot 13.7x) kept only so
# the thresholds in TCNT_RULES have a static, auditable copy.
# ---------------------------------------------------------------------------
BAND_SNAPSHOT_52W_LOW_PE: float = 12.6
BAND_SNAPSHOT_52W_HIGH_PE: float = 20.9

#: multiplier applied to EXIT_BASELINE_PCT inside each band
BAND_MULTIPLIERS: tuple[tuple[str, float, float], ...] = (
    # (label, upper bound as fraction of the 52w range, multiplier)
    ("下1/3 减半", 1 / 3, 0.5),
    ("中1/3 基线", 2 / 3, 1.0),
    ("上1/3 加速", 1.0, 2.0),
)
#: absolute multiple rails that override the percentile bands
PE_CLEAR_HALF: float = 23.0    # sell half of whatever remains
PE_CLEAR_ALL: float = 26.0     # ~10-year mean 25.73x — clear the rest
PE_DEEP_BRAKE: float = 10.0    # skip this tranche, drawing on SKIP_BUDGET

# ---------------------------------------------------------------------------
# Prefill fundamentals (quarterly manual entry; dashboard is the only writer)
# All figures RMB unless the field name says HKD.
# ---------------------------------------------------------------------------
DEFAULT_FUNDAMENTALS: tuple[dict, ...] = (
    {
        "quarter": "2026Q2",
        "non_ifrs_profit_billions": 68.4,
        "non_ifrs_profit_yoy_pct": 9.0,
        "marketing_services_yoy_pct": 22.0,
        "free_cash_flow_billions": -13.8,
        "net_cash_billions": 58.2,
        "buyback_hkd_billions": 12.2,   # H1 HK$24.4B, halved to a quarterly rate
        "notes": "2026-08-12 发布。营收 RMB2,048亿 +11%, 毛利率58%, 经营利润+12%。"
                 "IFRS归母 +0.7% 但非IFRS +9% —— 差额主要为投资重估/SBC/摊销, "
                 "经营利润才是干净读数。本土游戏 +17%(加速), 营销服务 +22%(AI广告模型驱动), "
                 "国际游戏 -0.8%, 社交网络 +0.8%, QQ MAU -2%。"
                 "⚠️ 自由现金流 -138亿, 2005年以来首次转负(剔除预付款后 +376亿); "
                 "资本开支 528亿 同比+176%/环比+65%; 净现金 582亿 同比-22%。"
                 "上半年回购 HK$244亿/5,003万股已注销, 约占港股回购总额1/4 —— "
                 "但回购在花资产负债表而非现金流, 这正是净现金-22%的来源。",
    },
)

# ---------------------------------------------------------------------------
# Rules. Every rule either PACES the exit or DECLARES the thesis broken.
# None of them can produce a buy.
# ---------------------------------------------------------------------------
TCNT_RULES: tuple[AlertRule, ...] = (
    # --- pacing: accelerate into strength -------------------------------
    AlertRule(
        rule_id="tcnt_exit_accelerate_upper_band",
        metric_key="tcnt.forward_pe",
        dimensions={"ticker": "0700.HK"},
        comparator="gt_consecutive",
        threshold=18.2,   # upper third of the design-date 52w range
        inclusive=True,
        max_gap_days=14,
        severity="advisory",
        action_label="进入52周区间上1/3 — 本档 ×2",
        description="Upper third of the trailing 52-week P/E range: double the "
                    "scheduled tranche. Recompute the band each quarter.",
    ),
    AlertRule(
        rule_id="tcnt_exit_clear_half_pe23",
        metric_key="tcnt.forward_pe",
        dimensions={"ticker": "0700.HK"},
        comparator="gt_consecutive",
        threshold=PE_CLEAR_HALF,
        inclusive=True,
        max_gap_days=14,
        severity="trigger",
        action_label="P/E ≥23x — 卖出剩余仓位的 50%",
    ),
    AlertRule(
        rule_id="tcnt_exit_clear_all_pe26",
        metric_key="tcnt.forward_pe",
        dimensions={"ticker": "0700.HK"},
        comparator="gt_consecutive",
        threshold=PE_CLEAR_ALL,
        inclusive=True,
        max_gap_days=14,
        severity="trigger",
        action_label="P/E ≥26x(≈10年均值) — 剩余全部清空, 不再等",
    ),
    # --- pacing: brake in deep value (budgeted) ---------------------------
    AlertRule(
        rule_id="tcnt_brake_deep_value_pe10",
        metric_key="tcnt.forward_pe",
        dimensions={"ticker": "0700.HK"},
        comparator="lt_consecutive",
        threshold=PE_DEEP_BRAKE,
        inclusive=True,
        max_gap_days=14,
        severity="advisory",
        action_label=f"P/E ≤10x — 可跳过本档(全程限 {SKIP_BUDGET} 次)",
        description="Deep-value brake. Draws on SKIP_BUDGET; skipped size is "
                    "redistributed so the completion date never slips.",
    ),
    # --- thesis invalidation: stop waiting for a better price ------------
    AlertRule(
        rule_id="tcnt_kill_fcf_negative_3q",
        metric_key="tcnt.fundamental",
        dimensions={"field": "free_cash_flow_billions"},
        comparator="lt_consecutive",
        threshold=0.0,
        consecutive_periods=3,
        max_gap_days=120,
        severity="trigger",
        action_label="自由现金流连续3季为负 — 全速清仓, 不再看价格",
        description="2026Q2 was the first negative FCF since 2005 (counter=1). "
                    "Ex-prepayments it was still +37.6B, so one quarter is a "
                    "timing question; three is structural.",
    ),
    AlertRule(
        rule_id="tcnt_kill_ads_below_10_2q",
        metric_key="tcnt.fundamental",
        dimensions={"field": "marketing_services_yoy_pct"},
        comparator="lt_consecutive",
        threshold=10.0,
        consecutive_periods=2,
        max_gap_days=120,
        severity="trigger",
        action_label="营销服务增速连续2季<10% — AI变现论点破产, 全速清仓",
        description="Marketing services (+22% in 2026Q2) is the ONLY hard "
                    "evidence that Tencent's AI spend monetises inside its own "
                    "P&L rather than being rented out. Lose it and the slow "
                    "schedule has no justification.",
    ),
    AlertRule(
        rule_id="tcnt_kill_profit_decline_2q",
        metric_key="tcnt.fundamental",
        dimensions={"field": "non_ifrs_profit_yoy_pct"},
        comparator="lt_consecutive",
        threshold=0.0,
        consecutive_periods=2,
        max_gap_days=120,
        severity="trigger",
        action_label="非IFRS归母净利连续2季同比负增长 — 全速清仓",
        description="The whole five-year case rests on ~9% EPS growth carrying "
                    "the exit price up even if the multiple never re-rates. "
                    "This rule watches that assumption directly.",
    ),
    AlertRule(
        rule_id="tcnt_kill_net_cash_negative",
        metric_key="tcnt.fundamental",
        dimensions={"field": "net_cash_billions"},
        comparator="lt_consecutive",
        threshold=0.0,
        max_gap_days=120,
        severity="trigger",
        action_label="净现金转负 — 全速清仓",
    ),
    # --- the buyback: support AND early warning ---------------------------
    AlertRule(
        rule_id="tcnt_buyback_cut_2q",
        metric_key="tcnt.fundamental",
        dimensions={"field": "buyback_hkd_billions"},
        comparator="lt_consecutive",
        threshold=8.0,
        consecutive_periods=2,
        max_gap_days=120,
        severity="trigger",
        action_label="季度回购<HK$80亿 连续2季 — 买盘消失+FCF确认, 加速清仓",
        description="H1 2026 ran HK$24.4B (~HK$12.2B/quarter, ~HK$500M/day) and "
                    "was ~1/4 of all HK-market buybacks. Crucially it is funded "
                    "from the balance sheet, not from FCF — net cash fell 22% "
                    "y/y. A cut is doubly bad: it removes the bid you are "
                    "selling into AND confirms the cash-flow squeeze.",
    ),
)

# ---------------------------------------------------------------------------
# Manual checklist
# ---------------------------------------------------------------------------
DEFAULT_CHECKLIST: tuple[dict, ...] = (
    {
        "id": "tcnt_quarterly_entry",
        "label": "季度财报录入 — 非IFRS归母净利/同比、营销服务增速、自由现金流、净现金、当季回购金额。"
                 "录入后倍数轨与52周区间自动重算, 判定本档卖多少",
        "url": "https://www.tencent.com/en-us/investors/financial-news.html",
        "cadence_days": 92,
    },
    {
        "id": "tcnt_buyback_pace",
        "label": "回购节奏月检 — 港交所每日回购披露。基准 HK$5亿/日、季度约HK$122亿。"
                 "缩量是双重利空(买盘消失 + FCF确认); 停止回购视同论点失效",
        "url": "https://www.hkexnews.hk/",
        "cadence_days": 31,
    },
    {
        "id": "tcnt_fcf_recovery",
        "label": "⭐自由现金流是否回正(当前计数 1/3) — Q2 -138亿为2005年以来首次。"
                 "剔除预付款后仍+376亿, 故一季属时点问题; Q3(约11月中)若仍为负则计数2, 距全速清仓一步",
        "url": "https://www.tencent.com/en-us/investors/financial-news.html",
        "cadence_days": 92,
    },
    {
        "id": "tcnt_regulatory",
        "label": "监管环境 — 游戏版号发放节奏、未成年人保护/反垄断新规、数据出境。"
                 "重大新监管行动直接触发全速清仓(2021年那轮是本框架不用SOTP定价的根本原因)",
        "url": "https://www.nppa.gov.cn/",
        "cadence_days": 31,
    },
    {
        "id": "tcnt_ai_monetization",
        "label": "AI变现验证 — 营销服务增速(当前+22%)、混元模型份额、WorkBuddy/CodeBuddy 商业化。"
                 "这是慢节奏方案的唯一理由: 腾讯的AI在自家广告系统里收钱, 不像美国超大厂的循环租赁",
        "url": "https://www.tencent.com/en-us/investors/financial-news.html",
        "cadence_days": 92,
    },
    {
        "id": "tcnt_investment_book",
        "label": "投资组合动向 — 减持/派发上市股权(京东/美团/Sea/Roblox等)是SOTP折价收敛的唯一现实路径。"
                 "大额实物派息会直接改变持仓性质, 需重估退出节奏",
        "url": "https://www.hkexnews.hk/",
        "cadence_days": 92,
    },
)
