"""Testable helper functions for the Portfolio dashboard page."""

from __future__ import annotations

from datetime import date

import polars as pl


def positions_to_trades(positions: list[dict]) -> list[dict]:
    """Convert MetadataStore positions to the trade format build_portfolio_values expects.

    Each position becomes a single BUY trade at avg_cost on date_acquired.
    Handles string dates from MetadataStore (e.g. '2023-01-15').
    """
    trades = []
    for p in positions:
        d = p["date_acquired"]
        if isinstance(d, str):
            d = date.fromisoformat(d)
        trades.append({
            "ticker": p["ticker"],
            "action": "BUY",
            "shares": p["shares"],
            "price": p["avg_cost"],
            "date": d,
        })
    return trades


def build_current_holdings(
    positions: list[dict],
    latest_prices: dict[str, float],
) -> dict[str, dict]:
    """Build {ticker: {shares, market_value}} for rebalancer.

    Falls back to shares * avg_cost if no price available.
    """
    result: dict[str, dict] = {}
    for p in positions:
        ticker = p["ticker"]
        shares = p["shares"]
        price = latest_prices.get(ticker, p.get("avg_cost", 0))
        result[ticker] = {
            "shares": shares,
            "market_value": shares * price,
        }
    return result


def build_risk_positions(
    positions: list[dict],
    latest_prices: dict[str, float],
    universe: list[dict],
) -> list[dict]:
    """Enrich positions with market_value and sector for risk analytics."""
    sector_map = {u["ticker"]: u.get("sector", "Unknown") for u in universe}
    result = []
    for p in positions:
        ticker = p["ticker"]
        shares = p["shares"]
        price = latest_prices.get(ticker, p.get("avg_cost", 0))
        result.append({
            "ticker": ticker,
            "shares": shares,
            "market_value": shares * price,
            "sector": sector_map.get(ticker, "Unknown"),
            "market": p.get("market", "us"),
        })
    return result


def build_portfolio_value_series(
    positions: list[dict],
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """Compute daily portfolio value directly from positions and prices.

    For each day, portfolio_value = sum(shares * close) across all held tickers.
    No dependency on trade dates — works as long as tickers have price history.

    Args:
        positions: list of dicts with at least {ticker, shares}
        prices: DataFrame with columns [ticker, date, close]

    Returns:
        DataFrame with columns [date, portfolio_value], sorted by date.
    """
    if not positions or prices.is_empty():
        return pl.DataFrame()

    holdings = {p["ticker"]: p["shares"] for p in positions}
    held_tickers = list(holdings.keys())

    # Filter prices to held tickers only
    held_prices = prices.filter(pl.col("ticker").is_in(held_tickers))
    if held_prices.is_empty():
        return pl.DataFrame()

    # Pivot: one column per ticker, rows are dates
    wide = held_prices.pivot(on="ticker", index="date", values="close").sort("date")

    # Compute portfolio value = sum(shares * price) for each day
    available = [t for t in held_tickers if t in wide.columns]
    if not available:
        return pl.DataFrame()

    value_expr = sum(pl.col(t) * holdings[t] for t in available)
    result = (
        wide.select(
            pl.col("date"),
            value_expr.alias("portfolio_value"),
        )
        .drop_nulls()
        .sort("date")
    )
    return result
