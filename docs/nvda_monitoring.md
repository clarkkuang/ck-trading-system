# NVDA 投资框架监控

三层买卖点信号系统,监控架构的第三个实例(前两个:[AI Model Share](ai_model_share_monitoring.md)、[AT&T/SpaceX](att_spacex_monitoring.md))。**规格书:[NVDA 分析报告](nvda_analysis_report.md)**。

## 定位

**中性论点评估** —— 持仓(1303 股 @ $20.85)仅作侧边栏参考,买卖阈值对称设置,不考虑成本。核心命题:*AI capex 周期还能走多长 × NVDA 份额与定价权守不守得住 × 当前价格付了多少倍*。

## 三层信号体系

| 层 | 数据 | 买点 | 卖点 |
|---|---|---|---|
| **技术**(周度自动) | 日线价格(yfinance) | 60 交易日高点回撤 ≥**12.7%** 且价>200 日线(门控) | 收盘连续 **4 周** <200 日线 |
| **估值**(半自动) | 季录 forward EPS,周算 P/E | P/E <**25x** | P/E >**45x** |
| **论点**(季度手动) | 财报录入 | — | DC QoQ 连续 2 季不升 / 指引<共识 / ASIC≥35% / 毛利<70% |

12.7% 来自 `strategies/portfolio_dip.py` 的 NVDA 校准(2019-2024 滚动 60 交易日回撤 75 分位)。

### 判定横幅
B=活跃买点数,S=活跃卖点数:B>0且S=0 → 🟢该买;S>0且B=0 → 🔴该卖/减;冲突 → 🟠人工裁决(**论点层优先于技术层**);否则 ⚪观望。

## 数据文件(data/monitoring/nvda/,git 跟踪)

| 文件 | 写者 | 内容 |
|---|---|---|
| prices.parquet | CI + backfill | **日线**收盘(与另两个监控的周线不同),dedupe(ticker,date),全历史 2016→今 |
| fundamentals.json | **仅 dashboard/backfill** | 季度 DC营收/QoQ/总营收/指引差/毛利率/ASIC份额/前瞻EPS |
| scenarios.json | **仅 dashboard/backfill** | 四档情景(bull 315/25%、base 240/40%、mild_bear 154/25%、structural_bear 105/10%,加权 $223.75) |
| alerts.json | CI + dashboard | 8 规则 episode 状态 |
| checklist.json | **仅 dashboard** | 6 项手动清单 |

## 财季标签约定(重要)

按**财季结束日所在的日历季**标注:FY27Q1(止 2026-04-26)→ `2026Q2`;FY26Q4 → `2026Q1`;**下次财报(~2026-08-26,FY27Q2)→ 录成 `2026Q3`**。财季号写在备注里。这样季度间隔恒 ≈90 天,规则的 120 天缺口语义正常工作。

## Runbook

```bash
# 一次性播种(必做——否则 CI 需 ~200 交易日才能算出 SMA200)
.venv/bin/python scripts/nvda_monitor_update.py --backfill

# 周度手动跑
.venv/bin/python scripts/nvda_monitor_update.py
```

- CI:`.github/workflows/nvda-monitor.yml`,周一 07:17 UTC(与 att 07:47、ai-share 06:17 错开);无需 secret
- Dashboard:页 10 "NVDA Framework";录季度数据/改 EPS 后规则**立即本地重评**
- 每季财报后(清单会提醒):录入 5 个数字(DC 营收、QoQ、指引差、毛利率、前瞻 EPS)+ 顺手更新 ASIC 份额(TrendForce)
- 改阈值:编辑 `monitoring/nvda_config.py`(阈值静态,不随情景编辑漂移)

## 已知边界

- **拆股**:yfinance 复权价;NVDA 再拆股后需删 prices.parquet 重新 `--backfill`
- P/E 在两次季度录入之间沿用旧 EPS(共识 EPS 本就低频)
- 2026Q1 的 DC 数字是估计值(notes 已标);ASIC 份额是第三方估计——触发后先核数再行动
- CI 断档 >30 天留数据洞,交易日滚动窗口会静默跨过
- SMH/TSM 历史从 2016 起;相对表现图各按自己首周对齐
