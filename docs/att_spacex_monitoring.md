# AT&T vs SpaceX 电信威胁监控

跟踪"SpaceX 进军电信对 AT&T 的结构性威胁"投资论点的周频监控系统。基于分析报告 v3(2026-07-03)。与 [AI Model Share Monitor](ai_model_share_monitoring.md) 同架构的第二个实例。

## 论点摘要

SpaceX 借 EchoStar 频谱交易(65 MHz,FCC 首次将全国性独占频谱交给卫星运营商)从"运营商供应商"升级为潜在直营竞争者。**市场定价的是 SpaceX 成功概率的边际变化,不是最终结果。**

两条路径:
- **A 批发/谈判筹码**(~50-55%):对 AT&T 影响有限
- **B 直营移动**(~35-40%):先蚕食农村,2028+ 威胁郊区

物理约束(报告 2.5 节):卫星 D2C 容量密度比地面网络低 **3-4 个数量级**,仅 <20 人/km² 区域可作主承载,100 人/km² 是硬顶。卫星射程内美国人口约 2-4%。

## 估值情景(可在 dashboard 编辑)

| 情景 | 默认概率 | 隐含价 | 带 |
|---|---:|---:|---|
| 基准: 威胁有限 | 45% | $28 | $26-30 |
| 温和悲观: 路径A + 防御性 CAPEX | 30% | $23 | $22-24 |
| 结构衰退: 路径B 兑现 | 20% | $16 | $15-17 |
| 极端尾部 | 5% | $12 | $11-13 |

加权公允 = **$23.30**。方法论:竞争加剧 → CAPEX 上升 → **FCF 率先下降** → 市场重定价(FCF 主轴,EV/EBITDA 交叉验证)。

## 监控指标 → 实现映射

### 自动化(周度,GitHub Actions)
- **T/VZ/TMUS/ASTS/SATS/SPCX 周收盘**(yfinance → `data/monitoring/att/prices.parquet`)
- 价格规则:T < $16(结构衰退带)/ T < $12(尾部带)→ 触发
- 展示:现价 vs 加权公允距离;六标的归一化相对表现(ASTS 上行=多寡头缓冲;SPCX=威胁情绪 proxy)

### 季度手动录入 → 自动规则(dashboard 表格,财报后 5 分钟)
| 规则 | 条件 | 动作 |
|---|---|---|
| churn ≥ 0.95%(含) | advisory | 复核路径B概率 |
| churn ≥ 1.10%(含)| trigger | 上调结构衰退概率 |
| churn 连升 3 季 | trigger | 结构性恶化确认 |
| 融合率连续 2 季走平/回落 | trigger | 捆绑见顶 |
| 季度 FCF < $4.0B | trigger | 空头获实证 |

阶梯语义:0.83%(基线)→ 0.95% → 1.10%;"市场通常提前约一年压低估值"。季度规则 `max_gap_days=120`(跳过一个季度即断 streak)。

### 手动清单(dashboard,带周期 Due 提醒)
1. **Starship V2 部署节奏**(月)— 威胁时间表总开关;2026 年底未稳定 → 整体后移 1-2 年
2. **终端生态**(季)— Qualcomm X105 2H26 出货;iPhone/三星 AWS-4/H-block 支持
3. **T-Mobile×SpaceX 走向**(月)— 续约=路径A / 谈崩=路径B
4. **Starlink 移动定价**(月)— $20-30=补充 / **$60+=正面替代**(T ARPU $85+)
5. **AT&T 季报录入**(季)— 下次 **Q2'26 约 7/22**,FCF 指引 $4.0-4.5B
6. **FCC/反垄断**(月)— 绩效义务执法、Warren 施压
7. **AST SpaceMobile**(月)— ASTS 起量 = D2D 多寡头 = 对 T 反而是缓冲

## 数据文件(data/monitoring/att/,git 跟踪)

| 文件 | 写者 | 内容 |
|---|---|---|
| prices.parquet | CI | 周收盘,dedupe(ticker, iso_week),每次拉 30 天日线自愈缺口 |
| alerts.json | CI + dashboard | 规则 episode 状态 |
| fundamentals.json | **仅 dashboard** | 季度 churn/融合率/FCF/Capex/净债务 |
| scenarios.json | **仅 dashboard** | 可编辑情景概率与价格带 |
| checklist.json | **仅 dashboard** | 手动清单 |

## Runbook

```bash
# 手动跑
.venv/bin/python scripts/att_monitor_update.py
# 调试
.venv/bin/python scripts/att_monitor_update.py --skip-prices --dry-run
```

- CI:`.github/workflows/att-monitor.yml`,周一 06:47 UTC(与 AI 监控错开 30 分钟避 push 竞争);无需任何 secret
- Dashboard:页面 09 "AT&T vs SpaceX";录入季度数据后规则**立即本地重评**,不必等 CI
- 改阈值/加规则:编辑 `monitoring/att_config.py`(阈值有意静态,不随情景编辑漂移)
- 规则触发后:看 episode 详情 → 按动作调整估值模型概率 → 用清单交叉验证

## 已知边界

- churn/融合率/FCF 依赖每季手动录入,不录则相应规则停留 insufficient_data
- 价格历史从部署日开始积累(首次约 4 周);SPCX 上市仅 3 周,相对表现图自其首周对齐
- 概率为主观估计;$9-11 为条件性尾部(四个坏事同时发生),报告 v2 已把公允悲观区间上修至 $14-17
