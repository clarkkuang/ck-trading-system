"""Stock screening API routes."""

from datetime import date

from fastapi import APIRouter, Depends

from ck_trading.api.deps import get_parquet_store
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


@router.get("/run/{strategy_name}")
async def run_screen(
    strategy_name: str,
    market: str = "us",
    store: ParquetStore = Depends(get_parquet_store),
):
    """Run a stock screen using the specified strategy."""
    strategy_class = STRATEGIES.get(strategy_name)
    if not strategy_class:
        return {"error": f"Unknown strategy: {strategy_name}"}

    strategy = strategy_class()
    prices = store.load_prices(market)
    fundamentals = store.load_fundamentals(market)

    if fundamentals.is_empty():
        return {"error": "No fundamental data available"}

    result = strategy.screen(prices, fundamentals, date.today())

    return {
        "strategy": strategy.name,
        "results": result.to_dicts() if not result.is_empty() else [],
        "count": result.height,
    }
