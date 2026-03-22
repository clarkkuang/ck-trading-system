"""FastAPI application for the trading system."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ck_trading.api.routes import backtest, market_data, portfolio, screening, signals


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    yield


app = FastAPI(
    title="CK Trading System",
    description="Quantitative value investing API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(market_data.router, prefix="/api/market", tags=["Market Data"])
app.include_router(signals.router, prefix="/api/signals", tags=["Signals"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtesting"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(screening.router, prefix="/api/screening", tags=["Screening"])


@app.get("/health")
async def health():
    return {"status": "ok"}
