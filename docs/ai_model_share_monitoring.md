# AI Model Share Monitor

评估开源/中国系模型对 Anthropic 竞争威胁的周频监控系统。把"侵蚀"叙事变成可计算的数字 + 触发阈值,盯方向和斜率,不盯绝对值。

## 三层指标体系

| 层 | 频率 | 内容 | 自动化 |
|---|---|---|---|
| **L1 先行** | 周 | OpenRouter 美元加权份额、npm/PyPI 下载曲线、旗舰降价检测 | ✅ 全自动(本系统) |
| **L2 验证** | 月 | Ramp AI Index、Artificial Analysis 价格-智能前沿、SWE-bench 差距 | 📋 手动清单 |
| **L3 确认** | 季/半年 | Menlo 半年报、a16z 调研、S-1/10-Q(NRR、Claude Code 收入占比) | 📋 手动清单 |

L2/L3 不自动化的原因:HTML 刮取易碎(页面改版即断),且频率低,人工核查成本可接受。Dashboard 上的清单带链接、周期和"上次核查日期",超期自动标 Due。

## 核心指标:美元加权份额

Token 份额会被便宜模型和 roleplay 流量污染(token 份额 51% 可能是噪音)。**美元份额过半才真正伤害论点**:

```
blended_usd_per_mtok = 0.70 × prompt_price + 0.30 × completion_price
dollar_volume(model, day) = tokens_total / 1e6 × blended
dollar_share(bloc, week) = Σ$(bloc) / Σ$(全部已定价、非 other 行)
```

- **70/30 是冻结常量**(`metrics.PROMPT_TOKEN_FRACTION`):真实 API 流量中 prompt token 占主导(对话历史、RAG、coding agent 传仓库)。改动此常量必须全序列重建 `weekly_bloc_share`(可从原始表完全重导)。
- 价格连接:as-of backward join(用 ≤ 当日的最新快照价);`:free` 变体按 $0 计(美元口径下正确);无价模型排除出美元分子分母,其 token 份额单独报告为测量误差。
- 周聚合:ISO 周,`days_observed ≥ 5` 才算完整周(`is_complete`)。

### 阵营(bloc)分类

`monitoring/blocs.py`,六分类:

| bloc | 说明 |
|---|---|
| `anthropic` | 论点主体,单列 |
| `western_closed` | OpenAI/Google/xAI/Cohere 等纯 API 实验室 |
| `western_open` | Meta/Mistral 开放权重/微调社区 |
| `chinese` | DeepSeek/Qwen/GLM/Kimi/MiniMax/字节/腾讯/百度/小米/快手… |
| `unclassified` | 未映射的新 org(日志告警,周度 review 时补表) |
| `other` | OpenRouter top-50 截断聚合行(不可归因的测量误差) |

两级优先:精确 model-id 覆盖(处理 Mistral 开/闭混合)→ org 前缀。`~org/...` 别名条目自动剥 `~` 归入真实 org。

### Scope 三态(编程类目口径)

Rankings API 的 `category=programming` 过滤支持未确认,每行数据带 scope 标注:

- `programming` — 响应回显了类目(确认生效)
- `programming:unverified` — 传参返回 200 但未回显(可能被静默忽略)
- `all` — 未传参

规则按 programming → unverified → all 顺序回退,单个评估窗口内不混用 scope。

## 触发规则(rules_config.py)

| rule_id | 条件 | 动作 |
|---|---|---|
| `or_cn_dollar_share_gt35_8w` | 中国系美元份额 >35% 连续 8 周 | **Bear 25%→30%** |
| `claude_opus_price_cut_gt30_365d` | Opus 旗舰 completion 价 365 天内跌 >30% | **Cut GM tier** |
| `claude_sonnet_price_cut_gt30_365d` | 同上 Sonnet | **Cut GM tier** |
| `claude_code_npm_decline_3w` | claude-code npm 连降 3 周且竞品 CLI 增长 | advisory |

### 精确语义

- **连续周判定**:从最新周向前数,值 >阈值 且相邻周间隔 ≤14 天(容忍恰好一次 CI 缺跑);更大缺口断 streak,缺失周不虚构为破线或未破线。最新周未破线 → 不触发。
- **降价判定**:滚动 365 天窗口内最高价 vs 当前;旗舰按 family 前缀取快照内 `created_at` 最新的模型 —— **更便宜的新旗舰替代旧旗舰也算有效降价**(设计意图)。`anthropic/*` 新模型不匹配任何 family 时日志告警提醒补 `FLAGSHIP_FAMILIES`。
- **Episode 模型**(alerts.json):触发 → 新 episode;持续为真 → 只更新 `last_period_key`;恢复 → `resolved_at`;同周重跑幂等。

## 数据文件(data/monitoring/,git 跟踪)

| 文件 | 内容 | 去重键 |
|---|---|---|
| `openrouter_daily_tokens.parquet` | 日度 top-50 token 量 | date+model_id+scope |
| `openrouter_pricing_snapshots.parquet` | 定价快照 | snapshot_date+model_id |
| `pkg_downloads.parquet` | npm/pypi 周下载 | registry+package+iso_week |
| `weekly_bloc_share.parquet` | 派生周度份额(每次全量重建) | — |
| `alerts.json` | 规则 episode 状态(canonical) | — |
| `checklist.json` | L2/L3 手动清单(**只有 dashboard 写**,CI 永不碰) | — |

`data/` 其余目录都被 gitignore,唯独 `data/monitoring/` 有意跟踪 —— CI 每周 commit 时间序列,本地 `git pull` 即同步告警状态。

## Runbook

### 首次配置

1. 注册免费 OpenRouter key: https://openrouter.ai/keys
2. 本地: `.env` 加 `OPENROUTER_API_KEY=sk-or-...`
3. CI: `gh secret set OPENROUTER_API_KEY --repo clarkkuang/ck-trading-system`
4. 无 key 也能跑:rankings 段跳过,pricing/npm/pypi 照常(美元加权份额无法计算)。

### 运行

```bash
# 手动跑一次
.venv/bin/python scripts/ai_share_update.py

# 调试选项
.venv/bin/python scripts/ai_share_update.py --skip-rankings --dry-run
```

- GitHub Action: `.github/workflows/ai-share-monitor.yml`,每周一 06:17 UTC 自动跑并 commit;也可在 Actions 页手动 `workflow_dispatch`。
- 退出码:0 = 成功或部分成功(个别段失败不阻断历史积累);1 = 全部采集段失败。
- 每段执行记录在 metadata.db 的 `job_runs` 表(`job_name=ai_share_update`)。

### 规则触发了怎么办

1. Dashboard(页面 08)看规则卡详情 + episode 历史;
2. 按触发表执行模型动作(如 Bear 权重上调);
3. 用 L2/L3 清单交叉验证(单一数据源不足为凭);
4. `alerts.json` 的 `context/details` 里有触发时的测量误差(other/unclassified 份额)。

### 维护

- **新 org 未分类**:运行日志和 `job_runs.result` 会列出 unclassified org → 在 `blocs.py` 的 `ORG_BLOC_MAP` 补一行。
- **Rankings API 变形**:解析集中在 `collectors/openrouter.py: _parse_rankings()`,一处修完;临时可用 `--skip-rankings` 或 workflow_dispatch 的 skip 输入。
- **改阈值/加规则**:只动 `monitoring/rules_config.py`。

## 已知局限

- OpenRouter ≠ 全市场:不含直连 API、企业合同、云市场流量 —— 它是**开发者边际流量**的 proxy,这正是先行指标想要的。
- `openai` SDK 下载被中国 OpenAI 兼容端点用户抬高,西方系下载被高估。
- 首次部署后 `or_cn_dollar_share_gt35_8w` 需要 ≥8 周数据积累,前两个月恒为 insufficient_data,属预期。
- 数据引用要求:*Source: OpenRouter (openrouter.ai/rankings), as of {date}*。
