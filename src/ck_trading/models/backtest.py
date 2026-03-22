"""Backtesting models."""

from dataclasses import dataclass, field
from datetime import date

import polars as pl


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    strategy_name: str
    universe: list[str]
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000
    max_positions: int = 20
    position_size_method: str = "equal_weight"  # or "score_weighted"
    rebalance_freq: str = "quarterly"  # "monthly", "quarterly", "annual"
    transaction_cost_bps: float = 10.0
    benchmark: str = "SPY"


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    config: BacktestConfig
    returns: pl.DataFrame  # columns: date, portfolio_return, benchmark_return
    trades: pl.DataFrame  # columns: date, ticker, action, shares, price, cost
    positions: pl.DataFrame  # columns: date, ticker, shares, value, weight
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Backtest: {self.config.strategy_name}"]
        lines.append(f"Period: {self.config.start_date} to {self.config.end_date}")
        lines.append(f"Initial Capital: ${self.config.initial_capital:,.0f}")
        for k, v in self.metrics.items():
            if "pct" in k or "return" in k or "ratio" in k:
                lines.append(f"  {k}: {v:.2%}")
            else:
                lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)
