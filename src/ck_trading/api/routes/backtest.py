"""Backtesting API routes."""

from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ck_trading.api.deps import get_parquet_store
from ck_trading.backtesting.engine import BacktestEngine
from ck_trading.models.backtest import BacktestConfig
from ck_trading.storage.parquet_store import ParquetStore
from ck_trading.strategies.composite_value import CompositeValueStrategy
from ck_trading.strategies.graham_defensive import GrahamDefensiveStrategy
from ck_trading.strategies.magic_formula import MagicFormulaStrategy
from ck_trading.strategies.piotroski_f_score import PiotroskiFScoreStrategy

router = APIRouter()

STRATEGIES = {
    "graham": GrahamDefensiveStrategy,
    "piotroski": PiotroskiFScoreStrategy,
    "magic_formula": MagicFormulaStrategy,
    "composite": CompositeValueStrategy,
}


class BacktestRequest(BaseModel):
    strategy: str = "graham"
    start_date: date = date(2015, 1, 1)
    end_date: date = date.today()
    initial_capital: float = 1_000_000
    max_positions: int = 20
    rebalance_freq: str = "quarterly"
    benchmark: str = "SPY"


@router.post("/run")
async def run_backtest(
    request: BacktestRequest,
    store: ParquetStore = Depends(get_parquet_store),
):
    """Run a backtest with the specified strategy."""
    strategy_class = STRATEGIES.get(request.strategy)
    if not strategy_class:
        return {"error": f"Unknown strategy: {request.strategy}"}

    strategy = strategy_class()
    prices = store.load_prices("us")
    fundamentals = store.load_fundamentals("us")

    if prices.is_empty():
        return {"error": "No price data available. Run backfill_data.py first."}

    # Get tickers from universe (exclude benchmarks from trading universe)
    tickers = prices["ticker"].unique().to_list()
    tickers = [t for t in tickers if t not in {"SPY", "QQQ", "IWM", "VTV"}]

    config = BacktestConfig(
        strategy_name=strategy.name,
        universe=tickers,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        max_positions=request.max_positions,
        rebalance_freq=request.rebalance_freq,
        benchmark=request.benchmark,
    )

    engine = BacktestEngine(strategy, config, prices, fundamentals)
    result = engine.run()

    return {
        "strategy": strategy.name,
        "metrics": result.metrics,
        "trades_count": result.trades.height,
        "summary": result.summary(),
    }
