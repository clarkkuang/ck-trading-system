# NFLX Framework Monitor (page 11)

第四个监控实例。论点:NFLX 处于"成长股→价值股"持有人换手期(Q2'26 财报确认:
增速 17.6→16.2→13.4→11.7% 指引,利润率 33.4%,史上最大回购 $4.7B/季)。

**预承诺买入阶梯**(2026-07 制定,详见对话分析与本页侧栏):
- T1 ≤$70 首批 $10-15K;T2 ≤$63 二批;趋势修复(重上 200 日线)= dip 门控重开
- 失效条件(季度录入触发):营收增速 <10% / 下季指引 <10% / 广告偏离 $3B 路径 / 利润率 <28%

数据:`data/monitoring/nflx/`(git 跟踪);周一 CI(nflx-monitor.yml, 07:47 UTC);
季度录入提醒:Q3 财报 ~10 月中(NFLX 财季=日历季);头号催化剂 = 1 月 FY27 指引。

播种:`scripts/nflx_monitor_update.py --backfill`(本地一次性)。
共享模块:technicals.py(本实例落地时从 nvda_metrics 提升)。
