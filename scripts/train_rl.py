#!/usr/bin/env python3
"""Train an RL (PPO) portfolio allocation agent.

Usage:
    .venv/bin/python scripts/train_rl.py \\
        --tickers NVDA AAPL MSFT GOOGL AMZN META NFLX TSLA QQQ TLT \\
        --train-start 2016-01-01 --train-end 2023-12-31 \\
        --val-start 2024-01-01 --val-end 2024-12-31 \\
        --total-timesteps 500000 \\
        --output models/rl_ppo.zip

Quick test (< 1 min):
    .venv/bin/python scripts/train_rl.py \\
        --tickers NVDA AAPL MSFT \\
        --total-timesteps 5000
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Train RL PPO portfolio agent")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument("--train-start", type=str, default="2016-01-01")
    parser.add_argument("--train-end", type=str, default="2023-12-31")
    parser.add_argument("--val-start", type=str, default="2024-01-01")
    parser.add_argument("--val-end", type=str, default="2024-12-31")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--reward", choices=["simple", "sharpe", "risk_adjusted"],
                        default="risk_adjusted")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--output", type=str, default="models/rl_ppo.zip")
    args = parser.parse_args()

    import csv

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ck_trading.rl.environment import TradingEnv
    from ck_trading.rl.features import FeatureBuilder
    from ck_trading.rl.reward import (
        RiskAdjustedReward,
        SharpeReward,
        SimpleReturnReward,
    )
    from ck_trading.storage.parquet_store import ParquetStore

    tickers = args.tickers
    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end)
    val_start = date.fromisoformat(args.val_start)
    val_end = date.fromisoformat(args.val_end)

    print(f"=== RL PPO Training ===")
    print(f"Tickers: {tickers}")
    print(f"Train: {train_start} → {train_end}")
    print(f"Val:   {val_start} → {val_end}")
    print(f"Timesteps: {args.total_timesteps:,}")
    print()

    # Load data
    print("Loading price data...", end=" ", flush=True)
    store = ParquetStore()
    prices = store.load_prices("us", tickers=tickers)
    fundamentals = store.load_fundamentals("us", tickers=tickers)
    print(f"Done. {prices.height:,} price rows, {fundamentals.height} fund rows.")

    if prices.is_empty():
        print("ERROR: No price data found. Run backfill first.")
        sys.exit(1)

    # Feature builder
    feature_builder = FeatureBuilder(tickers=tickers)

    # Reward function
    reward_map = {
        "simple": SimpleReturnReward(),
        "sharpe": SharpeReward(),
        "risk_adjusted": RiskAdjustedReward(),
    }
    reward_fn = reward_map[args.reward]
    print(f"Reward: {args.reward} ({reward_fn.__class__.__name__})")

    # Create training environment
    def make_train_env():
        return TradingEnv(
            prices=prices,
            fundamentals=fundamentals,
            tickers=tickers,
            start_date=train_start,
            end_date=train_end,
            feature_builder=feature_builder,
            reward_fn=reward_fn,
        )

    def make_val_env():
        return TradingEnv(
            prices=prices,
            fundamentals=fundamentals,
            tickers=tickers,
            start_date=val_start,
            end_date=val_end,
            feature_builder=feature_builder,
            reward_fn=reward_fn,
        )

    train_env = DummyVecEnv([make_train_env])
    val_env = DummyVecEnv([make_val_env])

    # PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=1,
    )

    print(f"\nObservation space: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space.shape}")
    print(f"Policy: MlpPolicy [64, 64]")
    print()

    # Callbacks
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Custom callback: record training metrics for visualization ---
    metrics_csv = output_dir / "training_metrics.csv"

    class MetricsCallback(BaseCallback):
        """Record training metrics to CSV for post-training visualization."""

        def __init__(self, csv_path: Path, verbose=0):
            super().__init__(verbose)
            self.csv_path = csv_path
            self._file = None
            self._writer = None

        def _on_training_start(self):
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow([
                "timesteps", "policy_loss", "value_loss", "entropy_loss",
                "approx_kl", "clip_fraction", "explained_variance",
            ])

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self):
            if self.logger and self._writer:
                logs = self.logger.name_to_value
                self._writer.writerow([
                    self.num_timesteps,
                    logs.get("train/policy_gradient_loss", ""),
                    logs.get("train/value_loss", ""),
                    logs.get("train/entropy_loss", ""),
                    logs.get("train/approx_kl", ""),
                    logs.get("train/clip_fraction", ""),
                    logs.get("train/explained_variance", ""),
                ])
                self._file.flush()

        def _on_training_end(self):
            if self._file:
                self._file.close()

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(output_dir / "best"),
        eval_freq=max(args.total_timesteps // 10, 1000),
        deterministic=True,
        verbose=1,
    )
    metrics_callback = MetricsCallback(metrics_csv)

    # Train
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, metrics_callback],
    )
    elapsed = time.time() - t0

    # Save model
    save_path = str(Path(args.output).with_suffix(""))  # SB3 adds .zip
    model.save(save_path)

    # Save config alongside model
    config_path = Path(args.output).with_suffix(".json")
    config = {
        "tickers": tickers,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "val_start": str(val_start),
        "val_end": str(val_end),
        "reward": args.reward,
        "lookback_window": feature_builder.lookback_window,
        "n_features": feature_builder.n_features,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Model: {args.output}")
    print(f"Config: {config_path}")
    print(f"Metrics: {metrics_csv}")

    # --- Generate training charts ---
    _generate_charts(metrics_csv, output_dir)

    print(f"\nTo use in screener/backtest:")
    print(f"  Select 'RL PPO' strategy in the dashboard")


def _generate_charts(metrics_csv: Path, output_dir: Path):
    """Generate training visualization charts from metrics CSV."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not metrics_csv.exists():
        print("No metrics CSV found, skipping charts.")
        return

    rows = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v else None
                except ValueError:
                    parsed[k] = None
            rows.append(parsed)

    if not rows:
        print("No training metrics recorded, skipping charts.")
        return

    timesteps = [r["timesteps"] for r in rows if r.get("timesteps") is not None]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Policy Loss", "Value Loss",
            "Entropy Loss", "KL Divergence",
            "Clip Fraction", "Explained Variance",
        ],
        vertical_spacing=0.08,
    )

    metrics = [
        ("policy_loss", 1, 1, "#1f77b4"),
        ("value_loss", 1, 2, "#ff7f0e"),
        ("entropy_loss", 2, 1, "#2ca02c"),
        ("approx_kl", 2, 2, "#d62728"),
        ("clip_fraction", 3, 1, "#9467bd"),
        ("explained_variance", 3, 2, "#8c564b"),
    ]

    for name, row, col, color in metrics:
        values = [r.get(name) for r in rows]
        valid_ts = [t for t, v in zip(timesteps, values) if v is not None]
        valid_vals = [v for v in values if v is not None]
        if valid_ts:
            fig.add_trace(
                go.Scatter(x=valid_ts, y=valid_vals, mode="lines",
                           name=name, line=dict(color=color)),
                row=row, col=col,
            )

    fig.update_layout(
        title="PPO Training Metrics",
        height=900, width=1000,
        showlegend=False,
    )
    for i in range(1, 7):
        fig.update_xaxes(title_text="Timesteps", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)

    chart_path = output_dir / "training_charts.html"
    fig.write_html(str(chart_path))
    print(f"Charts: {chart_path}")

    # Also generate a summary PNG-friendly chart (single combined)
    fig_summary = go.Figure()
    for name, _, _, color in metrics[:4]:  # top 4 metrics
        values = [r.get(name) for r in rows]
        valid_ts = [t for t, v in zip(timesteps, values) if v is not None]
        valid_vals = [v for v in values if v is not None]
        if valid_ts:
            fig_summary.add_trace(
                go.Scatter(x=valid_ts, y=valid_vals, mode="lines",
                           name=name.replace("_", " ").title(),
                           line=dict(color=color)),
            )
    fig_summary.update_layout(
        title="PPO Training Progress",
        xaxis_title="Timesteps",
        yaxis_title="Value",
        height=500,
    )
    summary_path = output_dir / "training_summary.html"
    fig_summary.write_html(str(summary_path))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
