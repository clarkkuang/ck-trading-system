"""Visualize the NVDA XGBoost demo: the overfitting trap, in charts.

Produces models/xgboost_demo.html with four panels:
  1. Train vs Test accuracy (the trap) — overfit vs regularized vs baseline
  2. Test AUC vs coin-flip line
  3. Feature importance
  4. "Trade on the signal" equity curve vs buy-and-hold (out-of-sample)

Run:
    .venv/bin/python scripts/xgboost_demo_viz.py
    open models/xgboost_demo.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import xgboost as xgb
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
NVDA = REPO / "data" / "prices" / "us" / "daily" / "NVDA.parquet"
OUT = REPO / "models" / "xgboost_demo.html"
HORIZON = 20


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("date")
    close = df["close"]
    feats = df.select("date", "close").with_columns([
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("ret_5d"),
        (pl.col("close") / pl.col("close").shift(10) - 1).alias("ret_10d"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("ret_20d"),
        (pl.col("close") / pl.col("close").shift(60) - 1).alias("ret_60d"),
        (pl.col("close") / pl.col("close").rolling_mean(20) - 1).alias("vs_sma20"),
        (pl.col("close") / pl.col("close").rolling_mean(50) - 1).alias("vs_sma50"),
        (pl.col("close") / pl.col("close").rolling_mean(200) - 1).alias("vs_sma200"),
        (pl.col("close").pct_change().rolling_std(20)).alias("vol_20d"),
        (1 - pl.col("close") / pl.col("close").rolling_max(60)).alias("dip_60d"),
    ])
    feats = feats.with_columns(
        (df["volume"] / df["volume"].rolling_mean(20)).alias("vol_ratio")
    )
    # forward 20d return + direction
    fwd_ret = (close.shift(-HORIZON) / close - 1).alias("fwd_ret")
    target = (close.shift(-HORIZON) > close).cast(pl.Int8).alias("target")
    feats = feats.with_columns([fwd_ret, target])
    return feats.drop_nulls()


def main() -> None:
    raw = pl.read_parquet(NVDA)
    feats = build_features(raw)
    feature_cols = [
        "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "vs_sma20", "vs_sma50", "vs_sma200",
        "vol_20d", "dip_60d", "vol_ratio",
    ]

    n = feats.height
    split = int(n * 0.70)
    train, test = feats[:split], feats[split:]
    X_tr = train.select(feature_cols).to_numpy()
    y_tr = train["target"].to_numpy()
    X_te = test.select(feature_cols).to_numpy()
    y_te = test["target"].to_numpy()

    # --- Model A: overfit ---
    overfit = xgb.XGBClassifier(
        n_estimators=500, max_depth=12, learning_rate=0.3,
        reg_lambda=0, reg_alpha=0, min_child_weight=1,
        subsample=1.0, colsample_bytree=1.0, eval_metric="logloss",
    )
    overfit.fit(X_tr, y_tr)
    a_tr = accuracy_score(y_tr, overfit.predict(X_tr))
    a_te = accuracy_score(y_te, overfit.predict(X_te))
    a_auc = roc_auc_score(y_te, overfit.predict_proba(X_te)[:, 1])

    # --- Model B: regularized ---
    reg = xgb.XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.03,
        reg_lambda=2.0, reg_alpha=0.5, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    reg.fit(X_tr, y_tr)
    b_tr = accuracy_score(y_tr, reg.predict(X_tr))
    b_te = accuracy_score(y_te, reg.predict(X_te))
    b_auc = roc_auc_score(y_te, reg.predict_proba(X_te)[:, 1])

    majority = int(round(y_tr.mean()))
    base_acc = accuracy_score(y_te, np.full_like(y_te, majority))

    # --- Trade-on-signal equity curves (out-of-sample) ---
    # Daily NVDA returns over the test window
    test_close = test["close"].to_numpy()
    daily_ret = np.concatenate([[0.0], test_close[1:] / test_close[:-1] - 1])

    # Buy & hold
    bh_equity = np.cumprod(1 + daily_ret)

    # Trade on overfit model: long when it predicts up, else flat (cash)
    sig_a = overfit.predict(X_te)
    a_equity = np.cumprod(1 + daily_ret * sig_a)

    # Trade on regularized model
    sig_b = reg.predict(X_te)
    b_equity = np.cumprod(1 + daily_ret * sig_b)

    test_dates = test["date"].to_list()

    # ============ Build figure ============
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "① 训练 vs 测试准确率 — 过拟合陷阱",
            "② 测试 AUC — 0.5 就是抛硬币",
            "③ 持仓信号实盘模拟 (样本外) vs 买入持有",
            "④ 因子重要性 (正则化模型)",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    # ① Train vs test accuracy grouped bars
    models = ["过拟合模型", "正则化模型", "无脑押多数类"]
    train_accs = [a_tr * 100, b_tr * 100, y_tr.mean() * 100]
    test_accs = [a_te * 100, b_te * 100, base_acc * 100]
    fig.add_trace(go.Bar(
        name="训练集", x=models, y=train_accs, marker_color="#94a3b8",
        text=[f"{v:.0f}%" for v in train_accs], textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="测试集", x=models, y=test_accs, marker_color="#2563eb",
        text=[f"{v:.0f}%" for v in test_accs], textposition="outside",
    ), row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="red",
                  annotation_text="抛硬币 50%", row=1, col=1)
    fig.update_yaxes(title_text="准确率 %", range=[0, 110], row=1, col=1)

    # ② AUC bars
    fig.add_trace(go.Bar(
        x=["过拟合模型", "正则化模型"], y=[a_auc, b_auc],
        marker_color=["#ef4444", "#f59e0b"],
        text=[f"{a_auc:.3f}", f"{b_auc:.3f}"], textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                  annotation_text="无预测力 0.5", row=1, col=2)
    fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=2)

    # ③ Equity curves
    fig.add_trace(go.Scatter(
        x=test_dates, y=bh_equity, name="买入持有 NVDA",
        line=dict(color="#16a34a", width=3),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=test_dates, y=b_equity, name="跟正则化模型信号交易",
        line=dict(color="#f59e0b", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=test_dates, y=a_equity, name="跟过拟合模型信号交易",
        line=dict(color="#ef4444", width=2, dash="dash"),
    ), row=2, col=1)
    fig.update_yaxes(title_text="累计净值 (起始=1)", row=2, col=1)

    # ④ Feature importance
    imp = reg.feature_importances_
    order = np.argsort(imp)
    fig.add_trace(go.Bar(
        x=[imp[i] for i in order],
        y=[feature_cols[i] for i in order],
        orientation="h", marker_color="#0891b2",
        text=[f"{imp[i]:.2f}" for i in order], textposition="outside",
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text="重要性", row=2, col=2)

    # Summary numbers for the title
    bh_final = (bh_equity[-1] - 1) * 100
    a_final = (a_equity[-1] - 1) * 100
    b_final = (b_equity[-1] - 1) * 100

    fig.update_layout(
        height=900, width=1300,
        title_text=(
            f"<b>XGBoost 预测 NVDA 20天涨跌 — 过拟合实证</b><br>"
            f"<sub>样本外 {test_dates[0]} → {test_dates[-1]}　|　"
            f"买入持有 {bh_final:+.0f}%　·　正则化信号 {b_final:+.0f}%　·　"
            f"过拟合信号 {a_final:+.0f}%</sub>"
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        margin=dict(t=90),
    )

    OUT.parent.mkdir(exist_ok=True)
    fig.write_html(str(OUT))
    print(f"Wrote {OUT}")
    print()
    print("Key numbers:")
    print(f"  Overfit:     train {a_tr:.1%} / test {a_te:.1%}  (gap {(a_tr-a_te)*100:.0f}pp), AUC {a_auc:.3f}")
    print(f"  Regularized: train {b_tr:.1%} / test {b_te:.1%}  (gap {(b_tr-b_te)*100:.0f}pp), AUC {b_auc:.3f}")
    print(f"  Baseline (always-up): test {base_acc:.1%}")
    print()
    print(f"  Out-of-sample equity:")
    print(f"    Buy & hold NVDA:        {bh_final:+.1f}%")
    print(f"    Trade on regularized:   {b_final:+.1f}%")
    print(f"    Trade on overfit:       {a_final:+.1f}%")


if __name__ == "__main__":
    main()
