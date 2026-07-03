"""XGBoost demo: predict NVDA 20-day forward direction (up/down).

Purpose: show BOTH how easy XGBoost is to use AND how dangerously
it overfits low-signal financial data. We train on the early years and
test on the held-out later years (proper time-series split — NO shuffling,
or we'd leak the future into the past).

Run:
    .venv/bin/python scripts/xgboost_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
NVDA = REPO / "data" / "prices" / "us" / "daily" / "NVDA.parquet"
HORIZON = 20  # predict direction 20 trading days ahead


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer a handful of classic technical features from OHLCV."""
    df = df.sort("date")
    close = df["close"]

    feats = df.select("date", "close").with_columns([
        # Momentum over various windows
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("ret_5d"),
        (pl.col("close") / pl.col("close").shift(10) - 1).alias("ret_10d"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("ret_20d"),
        (pl.col("close") / pl.col("close").shift(60) - 1).alias("ret_60d"),
        # Distance from moving averages
        (pl.col("close") / pl.col("close").rolling_mean(20) - 1).alias("vs_sma20"),
        (pl.col("close") / pl.col("close").rolling_mean(50) - 1).alias("vs_sma50"),
        (pl.col("close") / pl.col("close").rolling_mean(200) - 1).alias("vs_sma200"),
        # Realized volatility
        (pl.col("close").pct_change().rolling_std(20)).alias("vol_20d"),
        # Dip from rolling high
        (1 - pl.col("close") / pl.col("close").rolling_max(60)).alias("dip_60d"),
    ])

    # Volume features
    feats = feats.with_columns(
        (df["volume"] / df["volume"].rolling_mean(20)).alias("vol_ratio")
    )

    # TARGET: is the close 20 days FORWARD higher than today?  (1=up, 0=down)
    fwd = (close.shift(-HORIZON) > close).cast(pl.Int8).alias("target")
    feats = feats.with_columns(fwd)

    return feats.drop_nulls()


def main() -> None:
    raw = pl.read_parquet(NVDA)
    feats = build_features(raw)

    feature_cols = [
        "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "vs_sma20", "vs_sma50", "vs_sma200",
        "vol_20d", "dip_60d", "vol_ratio",
    ]

    # Time-series split: train on earlier 70%, test on later 30%. NO shuffle.
    n = feats.height
    split = int(n * 0.70)
    train = feats[:split]
    test = feats[split:]

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["target"].to_numpy()
    X_test = test.select(feature_cols).to_numpy()
    y_test = test["target"].to_numpy()

    print(f"NVDA samples: {n}  ({feats['date'][0]} → {feats['date'][-1]})")
    print(f"  train: {len(y_train)}  ({train['date'][0]} → {train['date'][-1]})")
    print(f"  test:  {len(y_test)}  ({test['date'][0]} → {test['date'][-1]})")
    print(f"  base rate (train up%): {y_train.mean():.1%}   test up%: {y_test.mean():.1%}")
    print()

    # ===== Model A: an INTENTIONALLY overfit model (deep trees, no regularization) =====
    print("=" * 64)
    print("Model A — OVERFIT on purpose (deep trees, 0 regularization)")
    print("=" * 64)
    overfit = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=12,          # very deep
        learning_rate=0.3,     # aggressive
        reg_lambda=0,          # no L2
        reg_alpha=0,           # no L1
        min_child_weight=1,    # allow tiny leaves
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric="logloss",
    )
    overfit.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, overfit.predict(X_train))
    te_acc = accuracy_score(y_test, overfit.predict(X_test))
    te_auc = roc_auc_score(y_test, overfit.predict_proba(X_test)[:, 1])
    print(f"  TRAIN accuracy: {tr_acc:.1%}   <- looks amazing")
    print(f"  TEST  accuracy: {te_acc:.1%}   <- reality")
    print(f"  TEST  AUC:      {te_auc:.3f}   (0.5 = coin flip)")
    print(f"  Overfit gap:    {(tr_acc - te_acc) * 100:.1f} pp")
    print()

    # ===== Model B: a regularized model (shallow, penalized) =====
    print("=" * 64)
    print("Model B — REGULARIZED (shallow trees, L1/L2, subsampling)")
    print("=" * 64)
    reg = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,           # shallow
        learning_rate=0.03,    # gentle
        reg_lambda=2.0,        # L2
        reg_alpha=0.5,         # L1
        min_child_weight=20,   # require big leaves
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    reg.fit(X_train, y_train)
    tr_acc_r = accuracy_score(y_train, reg.predict(X_train))
    te_acc_r = accuracy_score(y_test, reg.predict(X_test))
    te_auc_r = roc_auc_score(y_test, reg.predict_proba(X_test)[:, 1])
    print(f"  TRAIN accuracy: {tr_acc_r:.1%}")
    print(f"  TEST  accuracy: {te_acc_r:.1%}")
    print(f"  TEST  AUC:      {te_auc_r:.3f}")
    print(f"  Overfit gap:    {(tr_acc_r - te_acc_r) * 100:.1f} pp")
    print()

    # ===== Baseline: always predict the majority class =====
    majority = int(round(y_train.mean()))
    base_acc = accuracy_score(y_test, np.full_like(y_test, majority))
    print("=" * 64)
    print("Baselines")
    print("=" * 64)
    print(f"  Always-predict-'{majority}':  test accuracy {base_acc:.1%}")
    print(f"  Coin flip:                test accuracy ~50%")
    print()

    # ===== Feature importance (regularized model) =====
    print("=" * 64)
    print("Feature importance (regularized model)")
    print("=" * 64)
    imp = reg.feature_importances_
    for name, score in sorted(zip(feature_cols, imp), key=lambda x: -x[1]):
        bar = "█" * int(score * 60)
        print(f"  {name:12} {score:.3f} {bar}")


if __name__ == "__main__":
    main()
