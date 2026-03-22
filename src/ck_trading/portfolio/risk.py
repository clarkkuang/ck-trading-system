"""Portfolio risk metrics."""



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
