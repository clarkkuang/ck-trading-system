"""Market data API routes."""

from fastapi import APIRouter, Depends

from ck_trading.api.deps import get_parquet_store
from ck_trading.storage.parquet_store import ParquetStore

router = APIRouter()


@router.get("/prices/{market}/{ticker}")
async def get_prices(
    market: str,
    ticker: str,
    store: ParquetStore = Depends(get_parquet_store),
):
    """Get price history for a ticker."""
    df = store.load_prices(market, [ticker])
    if df.is_empty():
        return {"data": [], "count": 0}
    return {"data": df.to_dicts(), "count": df.height}


@router.get("/fundamentals/{market}/{ticker}")
async def get_fundamentals(
    market: str,
    ticker: str,
    store: ParquetStore = Depends(get_parquet_store),
):
    """Get fundamental data for a ticker."""
    df = store.load_fundamentals(market, [ticker])
    if df.is_empty():
        return {"data": [], "count": 0}
    return {"data": df.to_dicts(), "count": df.height}
