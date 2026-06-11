"""Dip Signals page — Portfolio-Tailored Buy-the-Dip status for each holding.

Shows, for every ticker in the user's portfolio:
    * current price + 60-day high
    * current dip% from that high
    * the ticker's calibrated dip threshold
    * 200-day SMA + whether price is in uptrend
    * a clear BUY / WATCH / HOLD verdict

This is the live, real-time view of the strategy implemented in
``ck_trading.strategies.portfolio_dip``.
"""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.dashboard.data_cache import load_prices_for_tickers
from ck_trading.portfolio.tracker import PortfolioTracker
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.storage.parquet_store import ParquetStore
from ck_trading.strategies.portfolio_dip import (
    DEFAULT_PORTFOLIO_THRESHOLDS,
    PortfolioDipStrategy,
)


def _refresh_holdings_prices(tickers: list[str], days_back: int = 14) -> dict:
    """Pull fresh prices for `tickers` via yfinance and save to ParquetStore.

    Returns a summary dict: {updated: [...], failed: [...], latest_date}.
    """
    if not tickers:
        return {"updated": [], "failed": [], "latest_date": None}

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=days_back)

    collector = USMarketCollector()
    store = ParquetStore()
    try:
        df = collector.collect_prices(tickers, start, end)
    except Exception as e:  # noqa: BLE001 — surface error to the UI
        return {
            "updated": [],
            "failed": tickers,
            "latest_date": None,
            "error": str(e),
        }

    if df.is_empty():
        return {"updated": [], "failed": tickers, "latest_date": None}

    # Drop rows with null/NaN close — yfinance returns a partial bar for the
    # current (not-yet-closed) trading day that pollutes downstream max()/min().
    df = df.filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
    if df.is_empty():
        return {"updated": [], "failed": tickers, "latest_date": None}

    store.save_prices(df, "us")
    updated = df["ticker"].unique().to_list()
    failed = [t for t in tickers if t not in set(updated)]
    latest = df["date"].max() if not df.is_empty() else None
    return {"updated": sorted(updated), "failed": sorted(failed), "latest_date": latest}

st.set_page_config(page_title="Dip Signals", page_icon="🎯", layout="wide")
st.title("Buy-the-Dip Signals")
st.caption(
    "Per-ticker dip thresholds calibrated to each holding's historical "
    "volatility (75th-percentile rolling 60-day drawdown, 2019–2024). "
    "A BUY signal fires when current dip ≥ threshold **and** price is "
    "above the 200-day SMA **and** the recent slide has stopped making "
    "new lows."
)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Signal Parameters")
    lookback = st.number_input(
        "Lookback (days)", min_value=20, max_value=252, value=60, step=5
    )
    reversal_window = st.number_input(
        "Reversal filter (days)", min_value=1, max_value=20, value=5,
        help="Today's close must be above the minimum of the last N closes.",
    )
    require_uptrend = st.checkbox(
        "Require uptrend (price > SMA200)", value=True,
        help="Disable to also signal during structural downtrends "
        "(e.g. catching INTC at its 2024 lows).",
    )
    trend_period = st.number_input(
        "Trend SMA period", min_value=50, max_value=400, value=200, step=10,
        disabled=not require_uptrend,
    )
    default_thr = st.slider(
        "Fallback threshold (unknown tickers)", 0.03, 0.30, 0.10, step=0.01,
        format="%.0f%%",
    )

    st.divider()
    st.markdown("**Calibrated thresholds:**")
    thresholds_df = pl.DataFrame({
        "ticker": list(DEFAULT_PORTFOLIO_THRESHOLDS.keys()),
        "threshold": list(DEFAULT_PORTFOLIO_THRESHOLDS.values()),
    }).sort("threshold")
    st.dataframe(
        thresholds_df.to_pandas(),
        use_container_width=True,
        hide_index=True,
        column_config={
            "threshold": st.column_config.NumberColumn("Min Dip", format="%.1f%%"),
        },
    )


# --------------------------------------------------------------------------
# Load portfolio + prices
# --------------------------------------------------------------------------
try:
    meta = MetadataStore()
    tracker = PortfolioTracker(meta)
    positions = tracker.get_positions()
    meta.close()
except Exception as e:
    st.error(f"Could not load portfolio: {e}")
    st.stop()

if not positions:
    st.warning(
        "No portfolio positions found. Add positions via the **Portfolio** "
        "page first."
    )
    st.stop()

us_tickers = sorted({
    p["ticker"] for p in positions
    if p.get("market", "US") == "US" and not p["ticker"].endswith(".HK")
})

if not us_tickers:
    st.warning("No US tickers in your portfolio.")
    st.stop()


# --------------------------------------------------------------------------
# Refresh data button — downloads latest prices for holdings via yfinance,
# clears the Streamlit data cache, and reruns the page.
# --------------------------------------------------------------------------
refresh_col, status_col = st.columns([1, 4])
with refresh_col:
    refresh_clicked = st.button(
        "🔄 Refresh data",
        help=(
            "Download the latest prices for your holdings and recompute "
            "signals. Takes a few seconds."
        ),
        type="primary",
        use_container_width=True,
    )
with status_col:
    if st.session_state.get("dip_refresh_result"):
        last = st.session_state["dip_refresh_result"]
        if last.get("error"):
            st.error(f"Refresh failed: {last['error']}")
        else:
            updated_n = len(last.get("updated", []))
            failed = last.get("failed", [])
            latest = last.get("latest_date")
            msg = f"✅ Refreshed {updated_n} tickers"
            if latest is not None:
                msg += f" — latest bar: **{latest}**"
            if failed:
                msg += f" · skipped: {', '.join(failed)}"
            st.success(msg)

if refresh_clicked:
    with st.spinner(f"Downloading latest prices for {len(us_tickers)} tickers…"):
        result = _refresh_holdings_prices(us_tickers)
    st.session_state["dip_refresh_result"] = result
    # Invalidate cached price loaders so the new bars show on the rerun
    try:
        load_prices_for_tickers.clear()
    except Exception:
        pass
    st.rerun()

prices = load_prices_for_tickers("us", tuple(us_tickers))
if prices.is_empty():
    st.error("No price data available for your holdings.")
    st.stop()

as_of = prices["date"].max()
st.caption(f"Data as-of: **{as_of}** · Tickers in portfolio: **{len(us_tickers)}**")


# --------------------------------------------------------------------------
# Compute per-ticker signal status
# --------------------------------------------------------------------------
def compute_status(ticker: str, df: pl.DataFrame) -> dict | None:
    # Drop rows with null/NaN close — yfinance can return a partial bar for
    # the current (not-yet-closed) day, which would break max()/min().
    closes = (
        df.sort("date")
        .filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
        ["close"]
        .to_list()
    )
    n = len(closes)
    if n < max(lookback, trend_period if require_uptrend else 0, reversal_window) + 1:
        return None
    current = closes[-1]
    window = closes[-lookback:]
    peak = max(window)
    if peak <= 0:
        return None
    dip_pct = (peak - current) / peak

    sma = sum(closes[-trend_period:]) / trend_period if require_uptrend else None
    in_uptrend = (current > sma) if sma is not None else True
    recent_low = min(closes[-reversal_window:]) if reversal_window > 1 else current
    stabilized = current > recent_low

    thr = DEFAULT_PORTFOLIO_THRESHOLDS.get(ticker, default_thr)

    # Verdict logic
    dip_passes = dip_pct >= thr
    if dip_passes and in_uptrend and stabilized:
        verdict = "BUY"
    elif dip_passes and not in_uptrend:
        verdict = "BLOCKED (downtrend)"
    elif dip_passes and not stabilized:
        verdict = "WATCH (still falling)"
    elif dip_pct >= thr * 0.7:
        verdict = "NEAR"
    else:
        verdict = "HOLD"

    return {
        "ticker": ticker,
        "current": current,
        "high_60d": peak,
        "dip_pct": dip_pct,
        "threshold": thr,
        "dip_vs_threshold": dip_pct / thr if thr > 0 else 0,
        "sma": sma,
        "in_uptrend": in_uptrend,
        "stabilized": stabilized,
        "verdict": verdict,
    }


rows = []
for t in us_tickers:
    df = prices.filter(pl.col("ticker") == t)
    if df.is_empty():
        continue
    status = compute_status(t, df)
    if status is None:
        rows.append({
            "ticker": t,
            "current": None,
            "high_60d": None,
            "dip_pct": None,
            "threshold": DEFAULT_PORTFOLIO_THRESHOLDS.get(t, default_thr),
            "dip_vs_threshold": None,
            "sma": None,
            "in_uptrend": None,
            "stabilized": None,
            "verdict": "INSUFFICIENT DATA",
        })
    else:
        rows.append(status)


# --------------------------------------------------------------------------
# Verdict summary cards
# --------------------------------------------------------------------------
buys = [r for r in rows if r["verdict"] == "BUY"]
watches = [r for r in rows if "WATCH" in r["verdict"]]
nears = [r for r in rows if r["verdict"] == "NEAR"]
blocked = [r for r in rows if "BLOCKED" in r["verdict"]]
holds = [r for r in rows if r["verdict"] == "HOLD"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🟢 BUY", len(buys))
c2.metric("🟡 WATCH", len(watches))
c3.metric("🟠 NEAR", len(nears))
c4.metric("⚫ BLOCKED", len(blocked))
c5.metric("⚪ HOLD", len(holds))

if buys:
    st.success(
        "**BUY signals:** "
        + ", ".join(
            f"{b['ticker']} (dip {b['dip_pct']:.1%} ≥ {b['threshold']:.1%})"
            for b in buys
        )
    )

# --------------------------------------------------------------------------
# Detailed table
# --------------------------------------------------------------------------
st.subheader("Per-Holding Status")

VERDICT_COLOR = {
    "BUY": "#15803d",
    "WATCH (still falling)": "#ca8a04",
    "NEAR": "#ea580c",
    "BLOCKED (downtrend)": "#525252",
    "HOLD": "#737373",
    "INSUFFICIENT DATA": "#a3a3a3",
}


high_col = f"{lookback}d High"
sma_col = f"SMA{trend_period}" if require_uptrend else "SMA"


def _format_row(r):
    if r["current"] is None:
        return {
            "Ticker": r["ticker"],
            "Verdict": r["verdict"],
            "Price": "n/a",
            high_col: "n/a",
            "Dip": "n/a",
            "Threshold": f"{r['threshold']:.1%}",
            "Dip ÷ Thr": "n/a",
            sma_col: "n/a",
            "Trend": "n/a",
            "Reversal": "n/a",
        }
    return {
        "Ticker": r["ticker"],
        "Verdict": r["verdict"],
        "Price": f"${r['current']:.2f}",
        high_col: f"${r['high_60d']:.2f}",
        "Dip": f"{r['dip_pct']:.1%}",
        "Threshold": f"{r['threshold']:.1%}",
        "Dip ÷ Thr": f"{r['dip_vs_threshold']:.2f}×",
        sma_col: f"${r['sma']:.2f}" if r["sma"] is not None else "—",
        "Trend": "↑ up" if r["in_uptrend"] else "↓ down",
        "Reversal": "✓ stabilized" if r["stabilized"] else "✗ still down",
    }


# Sort: BUY first, then by dip_vs_threshold desc
verdict_rank = {
    "BUY": 0, "WATCH (still falling)": 1, "NEAR": 2,
    "BLOCKED (downtrend)": 3, "HOLD": 4, "INSUFFICIENT DATA": 5,
}
rows_sorted = sorted(
    rows,
    key=lambda r: (
        verdict_rank.get(r["verdict"], 99),
        -(r["dip_vs_threshold"] or 0),
    ),
)
table_data = [_format_row(r) for r in rows_sorted]
st.dataframe(table_data, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Visualization: dip vs. threshold horizontal bar
# --------------------------------------------------------------------------
st.subheader("Dip vs. Threshold")
plot_rows = [r for r in rows_sorted if r["dip_pct"] is not None]
if plot_rows:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[r["ticker"] for r in plot_rows],
        x=[r["dip_pct"] * 100 for r in plot_rows],
        name="Current dip %",
        orientation="h",
        marker_color=[VERDICT_COLOR.get(r["verdict"], "#737373") for r in plot_rows],
        text=[f"{r['dip_pct']:.1%}" for r in plot_rows],
        textposition="auto",
    ))
    # Add threshold markers
    fig.add_trace(go.Scatter(
        y=[r["ticker"] for r in plot_rows],
        x=[r["threshold"] * 100 for r in plot_rows],
        mode="markers",
        name="Threshold",
        marker=dict(symbol="line-ns-open", size=20, color="#dc2626", line=dict(width=3)),
    ))
    fig.update_layout(
        height=max(300, 30 * len(plot_rows)),
        xaxis_title=f"Dip from {lookback}-day high (%)",
        yaxis_title="",
        showlegend=True,
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Per-ticker drill-down
# --------------------------------------------------------------------------
st.subheader("Drill-Down")
select_options = [r["ticker"] for r in rows_sorted if r["current"] is not None]
if select_options:
    ticker = st.selectbox("Inspect ticker", select_options, key="dip_drilldown")
    detail = next((r for r in rows if r["ticker"] == ticker), None)
    if detail is not None:
        # Price + 60d high + SMA chart
        df = prices.filter(pl.col("ticker") == ticker).sort("date")
        # Show last 1 year
        cutoff = as_of - timedelta(days=365)
        df_show = df.filter(pl.col("date") >= cutoff)
        if not df_show.is_empty():
            closes = df["close"].to_list()
            dates_list = df["date"].to_list()
            # Compute rolling 60d high and SMA200 series for plotting
            rolling_high = []
            rolling_sma = []
            for i in range(len(closes)):
                if i + 1 >= lookback:
                    rolling_high.append(max(closes[i + 1 - lookback : i + 1]))
                else:
                    rolling_high.append(None)
                if i + 1 >= trend_period:
                    rolling_sma.append(sum(closes[i + 1 - trend_period : i + 1]) / trend_period)
                else:
                    rolling_sma.append(None)

            # Slice to last year for display
            mask_start = next(
                (i for i, d in enumerate(dates_list) if d >= cutoff), 0
            )
            x_dates = dates_list[mask_start:]
            y_close = closes[mask_start:]
            y_high = rolling_high[mask_start:]
            y_sma = rolling_sma[mask_start:]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x_dates, y=y_close, name=f"{ticker} close",
                line=dict(color="#1f77b4", width=2),
            ))
            fig2.add_trace(go.Scatter(
                x=x_dates, y=y_high, name=f"{lookback}d high",
                line=dict(color="#16a34a", width=1, dash="dot"),
            ))
            if require_uptrend:
                fig2.add_trace(go.Scatter(
                    x=x_dates, y=y_sma, name=f"SMA{trend_period}",
                    line=dict(color="#dc2626", width=1, dash="dash"),
                ))
            # Mark current price
            fig2.add_trace(go.Scatter(
                x=[x_dates[-1]], y=[y_close[-1]],
                mode="markers", name="Current",
                marker=dict(size=12, color=VERDICT_COLOR.get(detail["verdict"], "#525252")),
                showlegend=False,
            ))
            fig2.update_layout(
                title=f"{ticker} — {detail['verdict']}",
                height=400,
                yaxis_tickformat="$.2f",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Buy zone summary
            thr = detail["threshold"]
            high60 = detail["high_60d"]
            buy_zone = high60 * (1 - thr)
            cur = detail["current"]
            distance = (cur - buy_zone) / cur * 100

            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Current", f"${cur:.2f}")
            cc2.metric(f"{lookback}d High", f"${high60:.2f}")
            cc3.metric("Buy zone trigger", f"≤ ${buy_zone:.2f}",
                       help=f"Threshold: {thr:.1%} dip from {lookback}d high")
            cc4.metric(
                "Distance to trigger",
                f"{distance:+.1f}%",
                delta=f"{cur - buy_zone:+.2f}",
                delta_color="inverse",
            )

            if detail["verdict"] == "BUY":
                st.success(
                    f"🟢 **BUY signal active.** Dip of {detail['dip_pct']:.1%} "
                    f"exceeds {ticker}'s calibrated threshold of {thr:.1%}, "
                    f"with price above SMA{trend_period} and recent stabilization."
                )
            elif "BLOCKED" in detail["verdict"]:
                st.warning(
                    f"⚫ **Dip is deep enough ({detail['dip_pct']:.1%}) but "
                    f"price is below SMA{trend_period}**. Strategy treats this "
                    f"as a structural downtrend, not a healthy pullback."
                )
            elif "WATCH" in detail["verdict"]:
                st.warning(
                    f"🟡 **Dip is deep enough but still making new lows.** "
                    f"Wait for the reversal filter (current > min of last "
                    f"{reversal_window} closes) before buying."
                )
            elif detail["verdict"] == "NEAR":
                st.info(
                    f"🟠 **{ticker} is approaching its trigger.** "
                    f"Current dip {detail['dip_pct']:.1%} vs. threshold {thr:.1%}. "
                    f"Needs to drop another {(buy_zone - cur) / cur * 100:+.1f}%."
                )
            else:
                st.info(
                    f"⚪ **{ticker} is HOLD.** Dip {detail['dip_pct']:.1%} "
                    f"is far from the {thr:.1%} threshold."
                )
