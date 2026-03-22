"""Portfolio and position models."""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel

from ck_trading.models.market_data import Market


class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Position(BaseModel):
    """A current portfolio position."""

    ticker: str
    market: Market
    shares: float
    avg_cost: float
    date_acquired: date
    current_price: float | None = None

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def market_value(self) -> float | None:
        if self.current_price is None:
            return None
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float | None:
        mv = self.market_value
        if mv is None:
            return None
        return mv - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float | None:
        pnl = self.unrealized_pnl
        if pnl is None:
            return None
        return pnl / self.cost_basis


class Trade(BaseModel):
    """A completed trade record."""

    ticker: str
    market: Market
    action: TradeAction
    shares: float
    price: float
    date: date
    fees: float = 0.0
    notes: str = ""


class PortfolioSnapshot(BaseModel):
    """Portfolio state at a point in time."""

    timestamp: datetime
    positions: list[Position]
    cash: float
    total_value: float
    daily_pnl: float | None = None
    total_pnl: float | None = None
