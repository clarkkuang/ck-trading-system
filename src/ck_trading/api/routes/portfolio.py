"""Portfolio API routes."""

from fastapi import APIRouter, Depends

from ck_trading.api.deps import get_metadata_store
from ck_trading.portfolio.tracker import PortfolioTracker
from ck_trading.storage.metadata_store import MetadataStore

router = APIRouter()


@router.get("/positions")
async def get_positions(
    store: MetadataStore = Depends(get_metadata_store),
):
    """Get all open positions."""
    tracker = PortfolioTracker(store)
    positions = tracker.get_positions()
    return {"positions": positions, "count": len(positions)}


@router.get("/watchlist")
async def get_watchlist(
    store: MetadataStore = Depends(get_metadata_store),
):
    """Get watchlist."""
    watchlist = store.get_watchlist()
    return {"watchlist": watchlist, "count": len(watchlist)}
