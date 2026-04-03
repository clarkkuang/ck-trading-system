"""Portfolio rebalance recommender.

Compare current holdings against target weights and recommend specific trades
to realign the portfolio.
"""


def calculate_drift(
    current: dict[str, dict],
    target_weights: dict[str, float],
) -> dict[str, float]:
    """Calculate weight drift for each ticker.

    Args:
        current: {ticker: {"shares": float, "market_value": float}}
        target_weights: {ticker: weight} where weights sum to ~1.0

    Returns:
        {ticker: drift} where drift = current_weight - target_weight.
        Positive = overweight, negative = underweight.
    """
    total_value = sum(p["market_value"] for p in current.values())
    all_tickers = set(list(current.keys()) + list(target_weights.keys()))

    drift: dict[str, float] = {}
    for ticker in all_tickers:
        current_weight = (
            current[ticker]["market_value"] / total_value
            if ticker in current and total_value > 0
            else 0.0
        )
        target_weight = target_weights.get(ticker, 0.0)
        drift[ticker] = current_weight - target_weight

    return drift


def recommend_rebalance_trades(
    current: dict[str, dict],
    target_weights: dict[str, float],
    prices: dict[str, float],
    total_value: float,
    min_trade_pct: float = 0.01,
) -> list[dict]:
    """Recommend specific trades to rebalance portfolio.

    Args:
        current: {ticker: {"shares": float, "market_value": float}}
        target_weights: {ticker: weight}
        prices: {ticker: current_price}
        total_value: total portfolio value (including cash)
        min_trade_pct: minimum trade size as % of portfolio to include

    Returns:
        list of {"ticker", "action", "shares", "estimated_value"} dicts
    """
    if not target_weights and not current:
        return []
    if total_value <= 0:
        return []

    drift = calculate_drift(current, target_weights)
    trades = []

    for ticker, d in drift.items():
        trade_value = abs(d) * total_value
        if abs(d) < min_trade_pct:
            continue

        price = prices.get(ticker, 0)
        if price <= 0:
            continue

        shares = int(trade_value / price)
        if shares <= 0:
            continue

        if d > 0:
            # Overweight — sell
            current_shares = current.get(ticker, {}).get("shares", 0)
            shares = min(shares, int(current_shares))
            if shares > 0:
                trades.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": shares,
                    "estimated_value": shares * price,
                })
        else:
            # Underweight — buy
            trades.append({
                "ticker": ticker,
                "action": "BUY",
                "shares": shares,
                "estimated_value": shares * price,
            })

    return sorted(trades, key=lambda t: t["estimated_value"], reverse=True)
