#!/usr/bin/env python3
"""Visualize RL training results in a human-friendly way.

Generates an interactive HTML report explaining:
1. What the model learned (portfolio weight allocation)
2. Training progress (is it still improving?)
3. Backtest comparison (RL vs equal-weight vs buy-and-hold)
4. Risk analysis (drawdown, volatility)

Usage:
    .venv/bin/python scripts/visualize_rl.py
    open models/rl_report.html
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    model_path = Path("models/rl_ppo.zip")
    config_path = Path("models/rl_ppo.json")
    metrics_path = Path("models/training_metrics.csv")

    if not model_path.exists():
        print("No trained model found. Run train_rl.py first.")
        sys.exit(1)

    # Load config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    tickers = config.get("tickers", [])
    if not tickers:
        print("No ticker config found.")
        sys.exit(1)

    print(f"Visualizing RL model for {len(tickers)} tickers: {tickers}")

    # Load model and data
    from stable_baselines3 import PPO
    from ck_trading.rl.features import FeatureBuilder
    from ck_trading.storage.parquet_store import ParquetStore

    store = ParquetStore()
    prices = store.load_prices("us", tickers=tickers)
    fundamentals = store.load_fundamentals("us", tickers=tickers)

    model = PPO.load(str(model_path))
    fb = FeatureBuilder(tickers=tickers)

    # --- 1. Current allocation recommendation ---
    latest_date = prices["date"].max()
    obs = fb.build_observation(prices, fundamentals, latest_date)
    action, _ = model.predict(obs, deterministic=True)
    exp_a = np.exp(action - np.max(action))
    weights = exp_a / exp_a.sum()

    # --- 2. Simulate backtest: RL vs equal-weight vs buy-and-hold ---
    import polars as pl

    test_start = date.fromisoformat(config.get("val_start", "2024-01-01"))
    test_end = latest_date if isinstance(latest_date, date) else date.today()

    # Get monthly rebalance dates in test period
    from ck_trading.rl.environment import _generate_monthly_dates
    rebal_dates = _generate_monthly_dates(test_start, test_end)

    # Simulate portfolios
    rl_values = [1.0]
    eq_values = [1.0]
    bnh_values = [1.0]  # buy-and-hold (first observation weights held forever)
    dates_out = [test_start]

    # Buy-and-hold: equal weight at start, never rebalance
    eq_w = np.ones(len(tickers)) / len(tickers)
    bnh_w = eq_w.copy()

    prev_rl_w = eq_w.copy()

    for i in range(len(rebal_dates) - 1):
        d_start = rebal_dates[i]
        d_end = rebal_dates[i + 1]

        # RL weights at this rebalance
        obs_i = fb.build_observation(prices, fundamentals, d_start)
        act_i, _ = model.predict(obs_i, deterministic=True)
        exp_i = np.exp(act_i - np.max(act_i))
        rl_w = exp_i / exp_i.sum()

        # Compute period returns per ticker
        period_returns = np.zeros(len(tickers))
        for j, t in enumerate(tickers):
            t_prices = prices.filter(
                (pl.col("ticker") == t) & (pl.col("date") >= d_start) & (pl.col("date") <= d_end)
            ).sort("date")
            if t_prices.height >= 2:
                period_returns[j] = t_prices["close"][-1] / t_prices["close"][0] - 1

        # Portfolio returns
        rl_ret = float(np.sum(rl_w * period_returns))
        eq_ret = float(np.sum(eq_w * period_returns))
        bnh_ret = float(np.sum(bnh_w * period_returns))

        rl_values.append(rl_values[-1] * (1 + rl_ret))
        eq_values.append(eq_values[-1] * (1 + eq_ret))
        bnh_values.append(bnh_values[-1] * (1 + bnh_ret))
        dates_out.append(d_end)

        prev_rl_w = rl_w

    # --- Build the report ---
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "📊 AI推荐的持仓比例 (当前)",
            "📈 回测对比: AI vs 等权 vs 买入持有",
            "📉 每月收益对比",
            "🎯 AI每次调仓的权重变化",
            "⚠️ 回撤对比",
            "📋 关键指标对比",
        ],
        specs=[
            [{"type": "pie"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # 1. Current allocation pie chart
    sorted_idx = np.argsort(weights)[::-1]
    pie_tickers = [tickers[i] for i in sorted_idx if weights[i] > 0.01]
    pie_weights = [float(weights[i]) for i in sorted_idx if weights[i] > 0.01]
    fig.add_trace(
        go.Pie(labels=pie_tickers, values=pie_weights,
               textinfo="label+percent", hole=0.3,
               marker=dict(colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                                   "#bcbd22", "#17becf"])),
        row=1, col=1,
    )

    # 2. Backtest comparison line chart
    fig.add_trace(
        go.Scatter(x=dates_out, y=rl_values, mode="lines",
                   name="🤖 AI (RL PPO)", line=dict(color="#1f77b4", width=3)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=dates_out, y=eq_values, mode="lines",
                   name="⚖️ 等权", line=dict(color="#ff7f0e", dash="dash")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=dates_out, y=bnh_values, mode="lines",
                   name="🛒 买入持有", line=dict(color="#2ca02c", dash="dot")),
        row=1, col=2,
    )

    # 3. Monthly returns bar chart
    rl_monthly = [rl_values[i + 1] / rl_values[i] - 1 for i in range(len(rl_values) - 1)]
    eq_monthly = [eq_values[i + 1] / eq_values[i] - 1 for i in range(len(eq_values) - 1)]
    bar_dates = dates_out[1:]
    fig.add_trace(
        go.Bar(x=bar_dates, y=rl_monthly, name="AI", marker_color="#1f77b4", opacity=0.7),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=bar_dates, y=eq_monthly, name="等权", marker_color="#ff7f0e", opacity=0.7),
        row=2, col=1,
    )

    # 4. Weight heatmap over time (RL weights at each rebalance)
    weight_matrix = []
    for i in range(len(rebal_dates) - 1):
        obs_i = fb.build_observation(prices, fundamentals, rebal_dates[i])
        act_i, _ = model.predict(obs_i, deterministic=True)
        exp_i = np.exp(act_i - np.max(act_i))
        w = exp_i / exp_i.sum()
        weight_matrix.append(w.tolist())

    if weight_matrix:
        fig.add_trace(
            go.Heatmap(
                z=np.array(weight_matrix).T,
                x=[str(d) for d in rebal_dates[:-1]],
                y=tickers,
                colorscale="Blues",
                showscale=True,
            ),
            row=2, col=2,
        )

    # 5. Drawdown comparison
    def calc_drawdown(values):
        peak = values[0]
        dd = []
        for v in values:
            peak = max(peak, v)
            dd.append(v / peak - 1)
        return dd

    rl_dd = calc_drawdown(rl_values)
    eq_dd = calc_drawdown(eq_values)
    fig.add_trace(
        go.Scatter(x=dates_out, y=rl_dd, mode="lines", fill="tozeroy",
                   name="AI 回撤", line=dict(color="#d62728")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates_out, y=eq_dd, mode="lines",
                   name="等权回撤", line=dict(color="#ff7f0e", dash="dash")),
        row=3, col=1,
    )

    # 6. Summary metrics table
    def total_return(vals):
        return vals[-1] / vals[0] - 1 if vals[0] > 0 else 0

    def max_drawdown(vals):
        dd = calc_drawdown(vals)
        return min(dd) if dd else 0

    def annualized_return(vals, n_months):
        if n_months <= 0 or vals[0] <= 0:
            return 0
        return (vals[-1] / vals[0]) ** (12 / n_months) - 1

    n_months = len(rl_values) - 1

    headers = ["指标", "🤖 AI (PPO)", "⚖️ 等权", "🛒 买入持有"]
    cells = [
        ["总收益", "年化收益", "最大回撤", "月度胜率"],
        [
            f"{total_return(rl_values)*100:.1f}%",
            f"{annualized_return(rl_values, n_months)*100:.1f}%",
            f"{max_drawdown(rl_values)*100:.1f}%",
            f"{sum(1 for r in rl_monthly if r > 0)/max(len(rl_monthly),1)*100:.0f}%",
        ],
        [
            f"{total_return(eq_values)*100:.1f}%",
            f"{annualized_return(eq_values, n_months)*100:.1f}%",
            f"{max_drawdown(eq_values)*100:.1f}%",
            f"{sum(1 for r in eq_monthly if r > 0)/max(len(eq_monthly),1)*100:.0f}%",
        ],
        [
            f"{total_return(bnh_values)*100:.1f}%",
            f"{annualized_return(bnh_values, n_months)*100:.1f}%",
            f"{max_drawdown(bnh_values)*100:.1f}%",
            f"—",
        ],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color="#1f77b4",
                        font=dict(color="white", size=14), align="center"),
            cells=dict(values=cells, fill_color=[["#f0f2f6"] * 4, ["#e8f4fd"] * 4,
                                                  ["#fff3e0"] * 4, ["#e8f5e9"] * 4],
                       font=dict(size=13), align="center", height=35),
        ),
        row=3, col=2,
    )

    fig.update_layout(
        title=dict(
            text="🤖 RL PPO 投资组合 — 训练结果报告",
            font=dict(size=22),
        ),
        height=1200, width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.7),
    )

    report_path = Path("models/rl_report.html")
    fig.write_html(str(report_path))
    print(f"\nReport saved: {report_path}")
    print("Open in browser: open models/rl_report.html")

    # Print summary to console too
    print(f"\n{'='*50}")
    print(f"🤖 RL PPO 模型摘要")
    print(f"{'='*50}")
    print(f"训练数据: {config.get('train_start')} → {config.get('train_end')}")
    print(f"测试数据: {config.get('val_start')} → {test_end}")
    print(f"股票池:   {', '.join(tickers)}")
    print(f"\n当前推荐持仓:")
    for i in sorted_idx:
        if weights[i] > 0.01:
            print(f"  {tickers[i]:6s}  {weights[i]*100:5.1f}%")
    print(f"\n回测对比 ({config.get('val_start')} → {test_end}):")
    print(f"  AI (PPO):  {total_return(rl_values)*100:+.1f}%  最大回撤 {max_drawdown(rl_values)*100:.1f}%")
    print(f"  等权:      {total_return(eq_values)*100:+.1f}%  最大回撤 {max_drawdown(eq_values)*100:.1f}%")
    print(f"  买入持有:  {total_return(bnh_values)*100:+.1f}%  最大回撤 {max_drawdown(bnh_values)*100:.1f}%")


if __name__ == "__main__":
    main()
