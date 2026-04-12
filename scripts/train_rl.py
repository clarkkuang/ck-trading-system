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

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
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

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(output_dir / "best"),
        eval_freq=max(args.total_timesteps // 10, 1000),
        deterministic=True,
        verbose=1,
    )

    # Train
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
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
    print(f"\nTo use in screener/backtest:")
    print(f"  Select 'RL PPO' strategy in the dashboard")


if __name__ == "__main__":
    main()
