# INTC Framework Monitor (page 12)

第五个监控实例。论点:Intel 是一个**二元反转期权** —— 代工能否成为可盈利的第三方
先进代工业务。与前四个(NVDA 成长 / NFLX 价值换手 / AT&T 防御 / AI-share)不同,
Intel 的核心变量是**离散**的,估值是**资产基础**(P/B + 分部,非前瞻 P/E —— GAAP 因
CHIPS 托管股按市值计价失真,Non-GAAP 前瞻 P/E ~60x)。完整框架见
[docs/intel_analysis_report.md](intel_analysis_report.md)。

**决定性变量(单一)**:代工在 18A-P/14A 先进节点上的**外部已承诺晶圆量 / 外部营收** ——
真实第三方业务(牛)vs 补贴型内部成本中心(熊/尾)。

**技术层 = 绝对价位阶梯**(NFLX 式,有意无视趋势门 —— 抛物线上涨使 NVDA 式"回调买入"
被永久锁死):T1 ≤$72 / T2 ≤$55 / T3 ≤$40。趋势修复(重上 200 日线)为"真跌破后夺回"预置。

**估值层 = P/B**:买 ≤2.0 / 减 ≥5.0(现 ~4.5x,Intel 历史最高)。

**论点层失效条件**:代工季亏 QoQ 走阔 / DCAI 增速连 2 季降 / AMD 服务器份额 ≥50% /
18A 年底 <70% 良率 / FCF 连 2 季负。

**初始判定(2026-07-23, ~$100)**:卖出观察区 / 择机减仓。加权公允 ~$83 vs 现价 ~$100,
已在牛市 SOTP(~$92)之上、创纪录 4.6x P/B —— 期权溢价已全额支付。基率诚实:国家冠军
反转完全重估约 1/3-1/4,成功非基准情形。

数据:`data/monitoring/intc/`(git 跟踪);周一 CI(intc-monitor.yml, 08:17 UTC);
季度录入提醒:Q3 财报 ~10 月(Intel 财季=日历季)。

播种:`scripts/intc_monitor_update.py --backfill`(本地一次性)。
共享模块:technicals.py(价格/技术);P/B 序列见 intc_metrics.pb_ratio_series。

**数据边界**:①18A 良率 Intel 官方从未披露(press 估 ~55-65%)—— 字段留空靠 checklist
人工跟踪 CFO "2026 年底" 红线,录入 press 估计会误触发卖点;②外部代工营收 Intel 不按季
清晰单列,规则把可自动化触发点放在 14A firm 客户数(离散)与代工季亏轨迹;③GAAP 全面
弃用,所有盈利/估值以 Non-GAAP + 分部经营利润 + P/B 为轴。
