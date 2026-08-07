"""User-editable configuration for the AT&T vs SpaceX telecom threat monitor.

This is the ONLY file to edit when tuning the monitor: watched tickers,
valuation scenario defaults, threshold rules, and the manual checklist.

Thesis (analysis report v3, 2026-07-03): SpaceX's EchoStar spectrum deal
(65 MHz, first exclusive nationwide D2D spectrum for a satellite operator)
upgrades it from carrier supplier to potential direct competitor. The market
prices the PROBABILITY of SpaceX success. Two paths: A wholesale/leverage
(~50-55%) vs B direct-to-consumer mobile (~35-40%). Physics constraint:
satellite D2C capacity density is 3-4 orders of magnitude below terrestrial
— primary-carrier viable only below ~20 people/km².

NOTE: rule thresholds ($16/$12 price bands, churn ladder, FCF floor) are
STATIC copies of the report's trigger table. They intentionally do NOT track
the editable scenarios.json — rule identity and episode history must stay
stable; editing display probabilities must never silently rewrite alert
semantics. To change a rule threshold, edit it here.
"""

from __future__ import annotations

from ck_trading.monitoring.rules import AlertRule

# ---------------------------------------------------------------------------
# Watched tickers: (ticker, role)
# ---------------------------------------------------------------------------
WATCH_TICKERS: tuple[tuple[str, str], ...] = (
    ("T", "subject"),           # AT&T — the thesis subject
    ("VZ", "peer"),             # Verizon
    ("TMUS", "peer"),           # T-Mobile (SpaceX's current D2D partner)
    ("ASTS", "buffer_signal"),  # AST SpaceMobile — rising = multi-oligopoly buffer
    ("SATS", "spectrum_signal"),  # EchoStar — spectrum deal counterparty
    ("SPCX", "threat_signal"),  # SpaceX (Nasdaq IPO 2026-06-12 @ $135)
)

ATT_DEDUPE_KEYS: dict[str, list[str]] = {"prices": ["ticker", "iso_week"]}

# ---------------------------------------------------------------------------
# Valuation scenarios (seed for scenarios.json; dashboard is the only writer)
# probability-weighted fair = 0.45*28 + 0.30*23 + 0.20*16 + 0.05*12 = 23.30
# ---------------------------------------------------------------------------
DEFAULT_SCENARIOS: tuple[dict, ...] = (
    {
        "id": "base",
        "label": "基准: SpaceX 威胁有限,指引兑现",
        "probability_pct": 45.0,
        "implied_price": 28.0,
        "price_low": 26.0,
        "price_high": 30.0,
    },
    {
        "id": "mild_bear",
        "label": "温和悲观: 路径A + 防御性 CAPEX 温和增加",
        "probability_pct": 30.0,
        "implied_price": 23.0,
        "price_low": 22.0,
        "price_high": 24.0,
    },
    {
        "id": "structural_decline",
        "label": "结构衰退: 路径B 兑现,农村流失 + CAPEX 大增",
        "probability_pct": 20.0,
        "implied_price": 16.0,
        "price_low": 15.0,
        "price_high": 17.0,
    },
    {
        "id": "extreme_tail",
        "label": "极端尾部: 直营成功+光纤停滞+去杠杆失败",
        "probability_pct": 5.0,
        "implied_price": 12.0,
        "price_low": 11.0,
        "price_high": 13.0,
    },
)

# ---------------------------------------------------------------------------
# Threshold rules (report section 3 trigger table)
# Quarterly rules use max_gap_days=120: adjacent quarter starts are 90-92
# days apart; a skipped quarter breaks streaks. Weekly rules keep default 14.
# ---------------------------------------------------------------------------
ATT_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        rule_id="att_price_below_16_structural",
        description="T 周收盘跌入结构衰退情景带 (<$16)",
        metric_key="att.price_weekly_close",
        dimensions={"ticker": "T"},
        comparator="lt_consecutive",
        threshold=16.0,
        consecutive_periods=1,
        action_label="论点恶化: 进入结构衰退带 ($15-17)",
        severity="trigger",
    ),
    AlertRule(
        rule_id="att_price_below_12_tail",
        description="T 周收盘跌入极端尾部带 (<$12)",
        metric_key="att.price_weekly_close",
        dimensions={"ticker": "T"},
        comparator="lt_consecutive",
        threshold=12.0,
        consecutive_periods=1,
        action_label="进入极端尾部带 ($11-13) — 复核论点/止损",
        severity="trigger",
    ),
    AlertRule(
        rule_id="att_churn_ge_095",
        description="后付费手机 churn ≥ 0.95%",
        metric_key="att.fundamental",
        dimensions={"field": "churn_pct"},
        comparator="gt_consecutive",
        threshold=0.95,
        inclusive=True,
        consecutive_periods=1,
        max_gap_days=120,
        action_label="流失爬坡 ≥0.95% — 复核路径B概率",
        severity="advisory",
    ),
    AlertRule(
        rule_id="att_churn_ge_110",
        description="后付费手机 churn ≥ 1.10%",
        metric_key="att.fundamental",
        dimensions={"field": "churn_pct"},
        comparator="gt_consecutive",
        threshold=1.10,
        inclusive=True,
        consecutive_periods=1,
        max_gap_days=120,
        action_label="流失警报 ≥1.10% — 上调结构衰退概率",
        severity="trigger",
    ),
    AlertRule(
        rule_id="att_churn_rising_3q",
        description="churn 连续 3 个季度环比上升",
        metric_key="att.fundamental",
        dimensions={"field": "churn_pct"},
        comparator="rising_streak",
        threshold=0.0,
        consecutive_periods=3,
        max_gap_days=120,
        action_label="流失连升3季 — 结构性恶化确认",
        severity="trigger",
    ),
    AlertRule(
        rule_id="att_convergence_stalled_2q",
        description="融合率连续 2 个季度走平或回落",
        metric_key="att.fundamental",
        dimensions={"field": "convergence_pct"},
        comparator="non_increasing_streak",
        threshold=0.0,
        consecutive_periods=2,
        max_gap_days=120,
        action_label="捆绑见顶 — 融合率连续两季走平/下降",
        severity="trigger",
    ),
    AlertRule(
        rule_id="att_fcf_below_4b",
        description="季度 FCF 低于 $4.0B 指引下限",
        metric_key="att.fundamental",
        dimensions={"field": "fcf_billions"},
        comparator="lt_consecutive",
        threshold=4.0,
        consecutive_periods=1,
        max_gap_days=120,
        action_label="空头获实证 — 季度FCF低于$4.0B",
        severity="trigger",
    ),
)

# ---------------------------------------------------------------------------
# Manual checklist (report section 3, prioritized indicators that cannot be
# automated; rendered by the dashboard, never evaluated as rules)
# ---------------------------------------------------------------------------
DEFAULT_CHECKLIST: tuple[dict, ...] = (
    {
        "id": "starship_v2_cadence",
        "label": "Starship V2 卫星部署节奏 — 威胁时间表总开关(2026年底未稳定部署→整体后移1-2年)",
        "url": "https://www.spacex.com/launches/",
        "cadence_days": 31,
    },
    {
        "id": "terminal_ecosystem",
        "label": "终端生态: Qualcomm X105 (2H26出货) / iPhone/三星 AWS-4/H-block 支持",
        "url": "https://www.qualcomm.com/news",
        "cadence_days": 92,
    },
    {
        "id": "tmobile_spacex_direction",
        "label": "SpaceX 路径判定 — 批发(路径A) vs 自营直销(路径B)。"
                 "⚠️[2026-08-06 重大修正] 路径B 已由 SpaceX 官方宣告, 不再是期权: 8/4 电话会 Shotwell 明确 SpaceX 将"
                 "自建美国地面移动网(用 EchoStar 交易内含的地面组件, 小型低成本基站配现有 Starlink 天线), 直接对标 T/VZ/TMUS, "
                 "原话: '我预计我们能抢到他们相当多的客户, 因为我认为我们的服务会更好'。"
                 "⚠️本规则原设计缺陷: 假定路径B会以'T-Mobile 谈崩'为信号——实际 SpaceX 是在保留批发合作的同时直接宣告自营, "
                 "触发条件设错。今后判定看'自营时间表与基站部署', 不看 T-Mobile 关系。"
                 "盯: 地面基站选址/铁塔协议、终端补贴、资费公布、T-Mobile 独家条款如何与自营并存",
        "url": "https://www.t-mobile.com/news",
        "cadence_days": 31,
    },
    {
        "id": "starlink_mobile_pricing",
        "label": "Starlink 移动定价 — $20-30=补充连接 / $60+=正面替代 (T 后付费 ARPU $85+)。"
                 "[2026-08-06] 当前所有 D2D 零售价 $0-10/月(T-Satellite $10、Docomo/KDDI 免费内含、Spark NZ $10)=互补区; "
                 "但那是'运营商批发'时代的价格, 自营 Starlink Mobile 资费尚未公布——这才是本项真正要等的数字。"
                 "⚠️时间表已提前: Shotwell 8/4 电话会称 V2 移动星 2027 年开始发射、'明年底开始提供服务'(=2027 年底), "
                 "容量口径 65MHz + 卫星数 10x → 自称'Starlink mobile 好 100 倍'。频谱 2027-11 完成过户。"
                 "我此前按 FCC 绩效节点(2029-11)推出的'2028-2030 才构成替代'是错的: 那是合规考核期限, 不是放号日期。"
                 "注意: SPCX 报表 ARPU $66 是宽带混合口径(国际 mix 致 $85→$66), 不是移动定价信号, 勿误读",
        "url": "https://www.starlink.com/",
        "cadence_days": 31,
    },
    {
        "id": "att_quarterly_earnings",
        "label": "AT&T 季报 — Q3'26 (~10月下旬): hard gate FCF ≈$4.9B(管理层Q3同比持平指引, 显著低于则$18B+路径重新紧绷); "
                 "融合率务必用有机口径(45→45已走平一格, 再平即触发年内首个黄灯); "
                 "电话会追问: 光纤存量提价机制(Q2被'artful recalibration'搪塞)",
        "url": "https://investors.att.com/",
        "cadence_days": 92,
    },
    {
        "id": "cband_spectrum_sept",
        "label": "C-band 频谱策略 — 9月董事会定调(Q2电话会Moffett之问被推迟); 关联: Q2未解释的频谱资产弃置费 + "
                 "EchoStar 频谱7月底交割后的杠杆路径(净债回2.5x承诺)",
        "url": "https://investors.att.com/",
        "cadence_days": 31,
    },
    {
        "id": "att_10q_check",
        "label": "10-Q 核对 (~8月初): 频谱弃置费的归因与金额(是否与D2D合资的频谱集中相关) + "
                 "净债/EBITDA 官方口径(监控里2.66x为估算) + EchoStar交割会计处理",
        "url": "https://investors.att.com/sec-filings",
        "cadence_days": 92,
    },
    {
        "id": "fcc_antitrust",
        "label": "FCC/反垄断: SpaceX 绩效义务执法、Warren 施压、频谱备案 (IBFS/ULS)。"
                 "[2026-08-06 核实] EchoStar 65MHz转让已于2026-05-12获批(局级令DA-26-471, 含地面部署灵活性豁免; 姊妹令同日批准T以~$23B购EchoStar 50MHz)。"
                 "硬节点: 频谱终割2027-11-30 / FCC首个绩效里程碑2029-11-30(SINR 5dB/70%可用/90% BEA)——路径B最早2028-2030才构成可用替代。"
                 "T/VZ/TMUS在docket零申报(不阻击、组JV对冲), 5/14三方D2D合资平台7月末仍未定稿",
        "url": "https://www.fcc.gov/news-events",
        "cadence_days": 31,
    },
    {
        "id": "asts_progress",
        "label": "AST SpaceMobile 进展 — ASTS 起量=D2D 多寡头而非 SpaceX 独占=对 T 反而是缓冲",
        "url": "https://ast-science.com/investors/",
        "cadence_days": 31,
    },
)
