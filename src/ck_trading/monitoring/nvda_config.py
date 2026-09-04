"""User-editable configuration for the NVDA investment framework monitor.

Spec: docs/nvda_analysis_report.md (三层买卖点体系). This is the ONLY file
to edit when tuning thresholds, tickers, scenarios, or the checklist.

Positioning: NEUTRAL thesis evaluation — the user holds 1303 shares @
$20.85 but thresholds are symmetric and ignore cost basis.

Three layers:
    1. Technical (auto, weekly): calibrated dip buy (12.7% from 60-trading-
       day high, gated by close>SMA200 — source: strategies/portfolio_dip.py
       NVDA calibration) + 200dma 4-week break sell.
    2. Valuation (semi-auto): user enters consensus forward EPS quarterly;
       forward P/E recomputed weekly from price. Buy <25x, sell >45x.
    3. Thesis (quarterly manual): DC QoQ decel 2q, guide miss, ASIC share
       >=35%, gross margin <70%.

Fiscal→calendar quarter labels: each NVDA fiscal quarter is labeled by the
calendar quarter containing its fiscal quarter-END. FY27Q1 (ended
2026-04-26) → "2026Q2"; FY26Q4 → "2026Q1"; next earnings (~2026-08-26,
FY27Q2) → enter as "2026Q3". Put the fiscal label in notes.

Rule thresholds are STATIC copies of the report's trigger table — editing
scenarios.json (display probabilities) never rewrites alert semantics.
"""

from __future__ import annotations

from ck_trading.monitoring.rules import AlertRule

# ---------------------------------------------------------------------------
# Watched tickers: (ticker, role)
# ---------------------------------------------------------------------------
WATCH_TICKERS: tuple[tuple[str, str], ...] = (
    ("NVDA", "subject"),
    ("AMD", "competitor"),
    ("AVGO", "competitor"),       # custom-ASIC proxy
    ("TSM", "supply_chain"),
    ("SMH", "sector_benchmark"),
)

NVDA_DEDUPE_KEYS: dict[str, list[str]] = {"prices": ["ticker", "date"]}

# Technical-layer constants (trading-day windows, matching the calibration)
DIP_LOOKBACK_DAYS = 60       # rolling high window (trading days)
SMA_PERIOD_DAYS = 200        # trend gate / break line
DIP_BUY_THRESHOLD = 0.127    # portfolio_dip.py NVDA 75th-pct calibration

# ---------------------------------------------------------------------------
# Valuation scenarios (seed for scenarios.json; dashboard-only writer after)
# weighted fair = .25*315 + .40*240 + .25*154 + .10*105 = 223.75
# ---------------------------------------------------------------------------
# 2026-09-03 REBUILD (not a re-weighting — the earnings base moved a decade).
#   Q2 FY27 (2026-08-26): revenue $96.2B +106% y/y, DC $89.0B +117%, GM 75.0%,
#   non-GAAP EPS $2.22 vs $2.09 consensus, Q3 guide $108B vs $104.2B consensus
#   (+4.0%), and Huang guided FY2028 revenue growth of ~70% on the call.
#   FY27E non-GAAP EPS is now ~$9.22 with three of four quarters visible.
#
#   The OLD table assumed EPS of $5.5-$9.0 — i.e. its most bullish case is
#   roughly where THIS YEAR already landed. Re-weighting a table whose EPS
#   axis is a year stale would have been meaningless, so the scenarios are
#   re-anchored on FY28E EPS (what the market actually prices) instead.
#
#   ⚠️ Honesty note on the result: this lands at a weighted fair value far
#   above spot, and the entire gap rests on FY28 growth >=50% AND the base
#   multiple holding 26x. At spot the market is paying roughly 20x on +25%
#   growth — i.e. it is pricing the MILD BEAR case, not the base. Treat the
#   upside as a statement about the assumptions, not a forecast.
# ---------------------------------------------------------------------------
DEFAULT_SCENARIOS: tuple[dict, ...] = (
    {
        "id": "bull",
        "label": "超级牛: Rubin全面放量+FY28增70%兑现+中国重开 (EPS $15.7×30x)",
        "probability_pct": 22.0,
        "implied_price": 470.0,
        "price_low": 430.0,
        "price_high": 510.0,
    },
    {
        "id": "base",
        "label": "基准: FY28增速降档至45-55%, 份额与毛利守住 (EPS $13.8×26x)",
        "probability_pct": 42.0,
        "implied_price": 360.0,
        "price_low": 330.0,
        "price_high": 390.0,
    },
    {
        "id": "mild_bear",
        "label": "温和熊: 增速降至25-30% + 倍数压缩至20x (EPS $11.5×20x)",
        "probability_pct": 27.0,
        "implied_price": 230.0,
        "price_low": 210.0,
        "price_high": 250.0,
    },
    {
        "id": "structural_bear",
        "label": "结构熊: 客户现金流/信用约束迫使capex减速+ASIC份额 (EPS $9.7×16x)",
        "probability_pct": 9.0,
        "implied_price": 155.0,
        "price_low": 140.0,
        "price_high": 175.0,
    },
)

# ---------------------------------------------------------------------------
# Prefill fundamentals (used by --backfill seeding; dashboard-only writer after)
# ---------------------------------------------------------------------------
DEFAULT_FUNDAMENTALS: tuple[dict, ...] = (
    {
        "quarter": "2026Q1",
        "dc_revenue_billions": 62.0,
        "dc_qoq_growth_pct": 16.0,
        "total_revenue_billions": None,
        "guide_vs_consensus_pct": None,
        "gross_margin_pct": None,
        "asic_server_share_pct": None,
        "forward_eps_consensus": None,
        "notes": (
            "FY26Q4(截至2026-01-25)。DC营收与QoQ为估计值(~$62B, ~+16%),"
            "仅用于让减速规则更早可评估;其余字段留空待补。"
        ),
    },
    {
        "quarter": "2026Q2",
        "dc_revenue_billions": 75.2,
        "dc_qoq_growth_pct": 21.0,
        "total_revenue_billions": 81.6,
        "guide_vs_consensus_pct": 4.6,
        "gross_margin_pct": 75.0,
        "asic_server_share_pct": 27.8,
        "forward_eps_consensus": 7.0,
        "ar_days": None,
        "customer_financing_exposure_pct": None,
        "notes": (
            "FY27Q1(截至2026-04-26, 2026-05-20发布)。营收+85% YoY/+20% QoQ;"
            "DC +92% YoY/+21% QoQ;Q2指引$91B vs 共识$87B(+4.6%);毛利率75.0%"
            "(non-GAAP);ASIC份额27.8%为TrendForce 2026全年预测;指引不含中国DC。"
            "信用通道字段(ar_days/敞口比)自8/26的10-Q起回填——2026-07-27循环融资"
            "抛售(OpenAI $250B担保谈判+SK $500B, NVDA CDS单日+14bp创纪录)后新增。"
        ),
    },
)

# ---------------------------------------------------------------------------
# Rules — nvda_buy_* / nvda_sell_* prefix is load-bearing: the dashboard
# splits card groups and computes the buy/sell verdict from it.
# ---------------------------------------------------------------------------
NVDA_RULES: tuple[AlertRule, ...] = (
    # ---- 买点 ----
    AlertRule(
        rule_id="nvda_buy_dip_gated",
        description="60交易日高点回撤≥12.7%(校准阈值)且价格在200日线上方",
        metric_key="nvda.gated_dip",
        dimensions={"ticker": "NVDA"},
        comparator="gt_consecutive",
        threshold=DIP_BUY_THRESHOLD,
        inclusive=True,
        consecutive_periods=1,
        action_label="技术买点: 上升趋势中回撤≥12.7% — 按纪律分批买入",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nvda_buy_pe_below_25",
        description="forward P/E < 25x(市场几乎不给加速定价)",
        metric_key="nvda.forward_pe",
        dimensions={"ticker": "NVDA"},
        comparator="lt_consecutive",
        threshold=25.0,
        consecutive_periods=1,
        action_label="估值买点: forward P/E<25x — 安全边际区",
        severity="advisory",
    ),
    # ---- 卖点 ----
    AlertRule(
        rule_id="nvda_sell_pe_above_45",
        description="forward P/E > 45x(需要'加速永续'假设)",
        metric_key="nvda.forward_pe",
        dimensions={"ticker": "NVDA"},
        comparator="gt_consecutive",
        threshold=45.0,
        consecutive_periods=1,
        action_label="估值卖点: forward P/E>45x — 历史减仓区",
        severity="advisory",
    ),
    AlertRule(
        rule_id="nvda_sell_below_200dma_4w",
        description="周收盘连续4周低于200日线",
        metric_key="nvda.pct_vs_200dma",
        dimensions={"ticker": "NVDA"},
        comparator="lt_consecutive",
        threshold=0.0,
        consecutive_periods=4,
        action_label="技术卖点: 连续4周低于200日线 — 趋势破位",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nvda_sell_dc_decel_2q",
        description="DC收入QoQ增速连续2季不升(平也算减速)",
        metric_key="nvda.fundamental",
        dimensions={"field": "dc_qoq_growth_pct"},
        comparator="non_increasing_streak",
        threshold=0.0,
        consecutive_periods=2,
        max_gap_days=120,
        action_label="论点卖点: DC增速连续两季不升 — 减速确认",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nvda_sell_guide_miss",
        description="指引 vs 共识 < 0(首次miss权重极大)",
        metric_key="nvda.fundamental",
        dimensions={"field": "guide_vs_consensus_pct"},
        comparator="lt_consecutive",
        threshold=0.0,
        consecutive_periods=1,
        max_gap_days=120,
        action_label="论点卖点: 指引低于共识 — 需求可见性转弱",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nvda_sell_asic_share_35",
        description="ASIC服务器份额 ≥ 35%(结构替代确认)",
        metric_key="nvda.fundamental",
        dimensions={"field": "asic_server_share_pct"},
        comparator="gt_consecutive",
        threshold=35.0,
        inclusive=True,
        consecutive_periods=1,
        max_gap_days=120,
        action_label="论点卖点: ASIC份额≥35% — 结构替代确认(先核数再行动)",
        severity="trigger",
    ),
    AlertRule(
        rule_id="nvda_sell_gm_below_70",
        description="毛利率 < 70%(定价权侵蚀,当前75%)",
        metric_key="nvda.fundamental",
        dimensions={"field": "gross_margin_pct"},
        comparator="lt_consecutive",
        threshold=70.0,
        consecutive_periods=1,
        max_gap_days=120,
        action_label="论点卖点: 毛利率<70% — 定价权侵蚀",
        severity="trigger",
    ),
    # ---- 信用通道(2026-07-27 循环融资抛售后新增) -------------------------
    AlertRule(
        rule_id="nvda_sell_ar_days_rising_2q",
        description="应收账款天数连续2季上升 — 循环融资的会计指纹(收入质量预警)",
        metric_key="nvda.fundamental",
        dimensions={"field": "ar_days"},
        comparator="rising_streak",
        threshold=0.0,
        consecutive_periods=2,
        max_gap_days=120,
        action_label="信用预警: AR天数连升2季 — 收入质量恶化, 核对10-Q应收明细",
        severity="advisory",
    ),
    AlertRule(
        rule_id="nvda_sell_financing_exposure_25",
        description="客户融资敞口(担保+客户股权+承诺)/TTM营收 ≥ 25% — 朗讯红线在20%+; "
                    "OpenAI $250B担保若签署将直接击穿",
        metric_key="nvda.fundamental",
        dimensions={"field": "customer_financing_exposure_pct"},
        comparator="gt_consecutive",
        threshold=25.0,
        inclusive=True,
        max_gap_days=120,
        action_label="信用卖点: 客户融资敞口≥25%营收 — vendor financing晚周期标记, 减仓重审",
        severity="trigger",
    ),
)

# ---------------------------------------------------------------------------
# Manual checklist
# ---------------------------------------------------------------------------
DEFAULT_CHECKLIST: tuple[dict, ...] = (
    {
        "id": "hyperscaler_capex",
        "label": "超大规模云厂商 capex 指引 — MSFT/GOOGL/AMZN/META 财报;盯2027是否兑现$1T",
        "url": "https://abc.xyz/investor/",
        "cadence_days": 92,
    },
    {
        "id": "rubin_ramp",
        "label": "Rubin 平台放量/供应链 — CoWoS、HBM4 产能,良率新闻",
        "url": "https://nvidianews.nvidia.com/",
        "cadence_days": 31,
    },
    {
        "id": "asic_competition",
        "label": "自研 ASIC 竞争: Trainium3/TPU v7/MI 系列采用度 → 更新 ASIC 份额字段",
        "url": "https://www.trendforce.com/news/",
        "cadence_days": 31,
    },
    {
        "id": "china_export",
        "label": "中国出口政策 — 指引已含零中国DC,政策回摆=纯上行期权",
        "url": "https://www.bis.doc.gov/",
        "cadence_days": 31,
    },
    {
        "id": "nvda_earnings_entry",
        "label": "NVDA 财报录入 — 下次 ~2026-08-26 (FY27Q2→录成2026Q3): DC营收/QoQ/指引差/毛利率/前瞻EPS "
                 "+ 信用通道两项: 10-Q承诺与或有事项(担保/客户股权→敞口比)、应收账款天数",
        "url": "https://investor.nvidia.com/",
        "cadence_days": 92,
    },
    {
        "id": "nvda_credit_conditions",
        "label": "信用条件月检(2026-07-27循环融资抛售后新增): NVDA 5Y CDS水平+周变速(预警: 单周+10bp或破100bp; "
                 "7/27曾单日+14bp至82bp创纪录) · Oracle/Meta CDS · AI债IG利差 · OpenAI/neocloud融资新闻 — "
                 "客户的资金成本=未来订单的贴现率; 数据源ICE无免费feed只能人工",
        "url": "https://www.bloomberg.com/quote/NVDA:US",
        "cadence_days": 31,
    },
    {
        "id": "hyperscaler_fcf",
        "label": "⭐客户自由现金流(本框架最大盲区, 2026-08-06 新增) — 五大 TTM FCF 与 capex/OCF 比。"
                 "机制: structural_bear 情景写的是'需求见顶', 但当前活的机制是"
                 "'客户 FCF 转负 → 被迫压 capex'。预警线: 任意两家 TTM FCF 同时为负, "
                 "或五家合计 TTM FCF 连续两季为负 → 上调 structural_bear 概率",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany",
        "cadence_days": 92,
    },
    {
        "id": "chip_cycle",
        "label": "芯片周期: TSMC 月度营收 / SEMI 北美设备 billings — 周期见顶前兆",
        "url": "https://pr.tsmc.com/english/quarterly-results",
        "cadence_days": 31,
    },
)
