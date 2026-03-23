"""Performance metrics for backtesting."""

import math

import polars as pl


def compute_metrics(
    returns: pl.DataFrame,
    initial_capital: float = 1_000_000,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
    num_trades: int | None = None,
) -> dict[str, float]:
    """Compute a suite of performance metrics from daily returns.

    Args:
        returns: DataFrame with columns [date, portfolio_return, benchmark_return]
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        trading_days: Trading days per year

    Returns:
        Dictionary of performance metrics
    """
    if returns.is_empty() or "portfolio_return" not in returns.columns:
        return {}

    rets = returns["portfolio_return"].to_list()
    if not rets:
        return {}

    # Cumulative returns
    cum = 1.0
    cum_values = []
    for r in rets:
        cum *= (1 + r)
        cum_values.append(cum)

    total_return = cum - 1
    n_days = len(rets)
    n_years = n_days / trading_days

    # CAGR
    if n_years > 0 and cum > 0:
        cagr = cum ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Volatility (annualized)
    if len(rets) > 1:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        daily_vol = math.sqrt(var)
        annual_vol = daily_vol * math.sqrt(trading_days)
    else:
        daily_vol = 0.0
        annual_vol = 0.0

    # Sharpe ratio
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    if daily_vol > 0:
        excess_returns = [r - daily_rf for r in rets]
        mean_excess = sum(excess_returns) / len(excess_returns)
        sharpe = mean_excess / daily_vol * math.sqrt(trading_days)
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation only)
    downside_rets = [min(r - daily_rf, 0) for r in rets]
    if downside_rets:
        downside_var = sum(r**2 for r in downside_rets) / len(downside_rets)
        downside_vol = math.sqrt(downside_var)
        if downside_vol > 0:
            mean_ret = sum(rets) / len(rets) - daily_rf
            sortino = mean_ret / downside_vol * math.sqrt(trading_days)
        else:
            sortino = 0.0
    else:
        sortino = 0.0

    # Maximum drawdown
    peak = cum_values[0]
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0
    for i, val in enumerate(cum_values):
        if val > peak:
            peak = val
            current_dd_start = i
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - current_dd_start

    # Win rate
    winning_days = sum(1 for r in rets if r > 0)
    total_trading_days = sum(1 for r in rets if r != 0)
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0.0

    # Average win / average loss
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Benchmark comparison
    bm_rets = returns["benchmark_return"].to_list() if "benchmark_return" in returns.columns else []
    if bm_rets:
        bm_cum = 1.0
        for r in bm_rets:
            bm_cum *= (1 + r)
        bm_total_return = bm_cum - 1
        bm_cagr = bm_cum ** (1 / n_years) - 1 if n_years > 0 and bm_cum > 0 else 0.0
    else:
        bm_total_return = 0.0
        bm_cagr = 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_days": max_dd_duration,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_trades": num_trades if num_trades is not None else total_trading_days,
        "benchmark_total_return": bm_total_return,
        "benchmark_cagr": bm_cagr,
        "alpha": cagr - bm_cagr,
        "final_value": initial_capital * cum,
    }
