"""Signal API routes."""

from fastapi import APIRouter, Depends

from ck_trading.api.deps import get_metadata_store
from ck_trading.storage.metadata_store import MetadataStore

router = APIRouter()


@router.get("/recent")
async def get_recent_signals(
    limit: int = 50,
    store: MetadataStore = Depends(get_metadata_store),
):
    """Get recent trading signals."""
    signals = store.get_recent_signals(limit)
    return {"signals": signals, "count": len(signals)}
