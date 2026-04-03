"""Portfolio risk metrics."""


import polars as pl


def concentration_analysis(positions: list[dict]) -> dict:
    """Analyze portfolio concentration by sector and market."""
    if not positions:
        return {"by_market": {}, "by_sector": {}, "hhi": 0.0, "top5_weight": 0.0}

    total_value = sum(p.get("market_value", 0) for p in positions)
    if total_value <= 0:
        return {"by_market": {}, "by_sector": {}, "hhi": 0.0, "top5_weight": 0.0}

    # By market
    by_market: dict[str, float] = {}
    for p in positions:
        market = p.get("market", "US")
        by_market[market] = by_market.get(market, 0) + p.get("market_value", 0)
    by_market = {k: v / total_value for k, v in by_market.items()}

    # By sector (if available)
    by_sector: dict[str, float] = {}
    for p in positions:
        sector = p.get("sector", "Unknown")
        by_sector[sector] = by_sector.get(sector, 0) + p.get("market_value", 0)
    by_sector = {k: v / total_value for k, v in by_sector.items()}

    # HHI (Herfindahl-Hirschman Index) - measures concentration
    weights = [p.get("market_value", 0) / total_value for p in positions]
    hhi = sum(w**2 for w in weights)

    # Top 5 concentration
    sorted_weights = sorted(weights, reverse=True)
    top5_weight = sum(sorted_weights[:5])

    return {
        "by_market": by_market,
        "by_sector": by_sector,
        "hhi": hhi,
        "top5_weight": top5_weight,
        "num_positions": len(positions),
    }


def correlation_matrix(
    prices: pl.DataFrame,
    tickers: list[str],
) -> pl.DataFrame:
    """Compute pairwise return correlation matrix.

    Args:
        prices: DataFrame with columns [ticker, date, close]
        tickers: list of tickers to include

    Returns:
        DataFrame where rows and columns are tickers, values are correlations.
    """
    if prices.is_empty() or not tickers:
        return pl.DataFrame()

    # Pivot to wide: one column per ticker with daily returns
    filtered = prices.filter(pl.col("ticker").is_in(tickers))
    if filtered.is_empty():
        return pl.DataFrame()

    wide = (
        filtered
        .sort("date")
        .pivot(on="ticker", index="date", values="close")
    )

    # Calculate daily returns for each ticker
    return_cols = []
    available_tickers = [t for t in tickers if t in wide.columns]
    for t in available_tickers:
        wide = wide.with_columns(
            (pl.col(t) / pl.col(t).shift(1) - 1).alias(f"_ret_{t}")
        )
        return_cols.append(f"_ret_{t}")

    # Drop first row (null from shift) and extract return columns
    returns_df = wide.drop_nulls(subset=return_cols).select(return_cols)

    # Compute correlation using Polars pearson_corr
    corr_data: dict[str, list[float]] = {t: [] for t in available_tickers}
    for i, ti in enumerate(available_tickers):
        for j, tj in enumerate(available_tickers):
            if i == j:
                corr_data[ti].append(1.0)
            else:
                corr_val = returns_df.select(
                    pl.corr(f"_ret_{ti}", f"_ret_{tj}").alias("c")
                )["c"][0]
                corr_data[ti].append(
                    float(corr_val) if corr_val is not None else 0.0
                )

    result = pl.DataFrame(corr_data)
    return result


def stress_test(
    positions: list[dict],
    scenarios: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """Run stress test scenarios on portfolio.

    Args:
        positions: list of dicts with ticker, market_value, sector
        scenarios: {scenario_name: {key: shock_pct}}
            key can be: "all", a sector name, or a specific ticker.
            shock_pct: e.g. -0.20 for a 20% drop.

    Returns:
        {scenario_name: {"dollar_impact", "portfolio_impact", "by_ticker"}}
    """
    total_value = sum(p.get("market_value", 0) for p in positions)
    results: dict[str, dict] = {}

    for scenario_name, shocks in scenarios.items():
        dollar_impact = 0.0
        by_ticker: dict[str, float] = {}

        for p in positions:
            ticker = p.get("ticker", "")
            sector = p.get("sector", "Unknown")
            mv = p.get("market_value", 0)

            # Priority: ticker-specific > sector > "all"
            if ticker in shocks:
                shock = shocks[ticker]
            elif sector in shocks:
                shock = shocks[sector]
            elif "all" in shocks:
                shock = shocks["all"]
            else:
                shock = 0.0

            impact = mv * shock
            dollar_impact += impact
            by_ticker[ticker] = impact

        results[scenario_name] = {
            "dollar_impact": dollar_impact,
            "portfolio_impact": dollar_impact / total_value if total_value else 0.0,
            "by_ticker": by_ticker,
        }

    return results


def portfolio_var(
    prices: pl.DataFrame,
    holdings: dict[str, float],
    confidence: float = 0.95,
) -> float:
    """Historical Value-at-Risk for a portfolio.

    Args:
        prices: DataFrame with columns [ticker, date, close]
        holdings: {ticker: market_value_in_dollars}
        confidence: confidence level (e.g. 0.95 for 95% VaR)

    Returns:
        VaR as a negative dollar amount (expected loss at confidence level).
        Returns 0.0 if insufficient data.
    """
    if not holdings or prices.is_empty():
        return 0.0

    tickers = list(holdings.keys())
    total_value = sum(holdings.values())
    if total_value <= 0:
        return 0.0

    # Get weighted portfolio returns
    filtered = prices.filter(pl.col("ticker").is_in(tickers)).sort("date")
    if filtered.is_empty():
        return 0.0

    wide = filtered.pivot(on="ticker", index="date", values="close")

    # Calculate daily portfolio return as weighted sum
    available = [t for t in tickers if t in wide.columns]
    if not available:
        return 0.0

    weights = {t: holdings[t] / total_value for t in available}

    # Compute individual returns
    for t in available:
        wide = wide.with_columns(
            (pl.col(t) / pl.col(t).shift(1) - 1).alias(f"_ret_{t}")
        )

    wide = wide.drop_nulls(subset=[f"_ret_{t}" for t in available])

    # Weighted portfolio return
    port_ret_expr = sum(
        pl.col(f"_ret_{t}") * weights[t] for t in available
    )
    wide = wide.with_columns(port_ret_expr.alias("_port_ret"))

    port_returns = wide["_port_ret"].sort().to_list()
    if not port_returns:
        return 0.0

    # VaR = percentile of losses
    idx = int((1 - confidence) * len(port_returns))
    idx = max(0, min(idx, len(port_returns) - 1))
    var_return = port_returns[idx]

    return var_return * total_value
