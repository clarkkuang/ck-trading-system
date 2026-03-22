"""Portfolio position tracking and P&L calculation."""

from datetime import date

from ck_trading.models.portfolio import Position, Trade, TradeAction
from ck_trading.storage.metadata_store import MetadataStore


class PortfolioTracker:
    """Track positions and calculate P&L."""

    def __init__(self, metadata_store: MetadataStore):
        self.store = metadata_store

    def add_trade(self, trade: Trade) -> int:
        """Record a trade and update positions."""
        trade_id = self.store.save_trade(trade)

        if trade.action == TradeAction.BUY:
            # Check if position exists
            positions = self.store.get_open_positions()
            existing = [p for p in positions if p["ticker"] == trade.ticker]

            if existing:
                # Average up/down
                pos = existing[0]
                total_shares = pos["shares"] + trade.shares
                total_cost = pos["shares"] * pos["avg_cost"] + trade.shares * trade.price
                avg_cost = total_cost / total_shares

                self.store.conn.execute(
                    "UPDATE positions SET shares = ?, avg_cost = ? WHERE id = ?",
                    (total_shares, avg_cost, pos["id"]),
                )
                self.store.conn.commit()
            else:
                position = Position(
                    ticker=trade.ticker,
                    market=trade.market,
                    shares=trade.shares,
                    avg_cost=trade.price,
                    date_acquired=trade.date,
                )
                self.store.save_position(position)

        elif trade.action == TradeAction.SELL:
            positions = self.store.get_open_positions()
            existing = [p for p in positions if p["ticker"] == trade.ticker]
            if existing:
                pos = existing[0]
                remaining = pos["shares"] - trade.shares
                if remaining <= 0:
                    self.store.conn.execute(
                        "UPDATE positions SET is_closed = 1, closed_at = ? WHERE id = ?",
                        (date.today().isoformat(), pos["id"]),
                    )
                else:
                    self.store.conn.execute(
                        "UPDATE positions SET shares = ? WHERE id = ?",
                        (remaining, pos["id"]),
                    )
                self.store.conn.commit()

        return trade_id

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        return self.store.get_open_positions()

    def calculate_portfolio_value(
        self, current_prices: dict[str, float]
    ) -> dict:
        """Calculate total portfolio value and P&L."""
        positions = self.get_positions()
        total_cost = 0.0
        total_value = 0.0
        position_details = []

        for pos in positions:
            ticker = pos["ticker"]
            shares = pos["shares"]
            avg_cost = pos["avg_cost"]
            cost = shares * avg_cost
            total_cost += cost

            price = current_prices.get(ticker, avg_cost)
            value = shares * price
            total_value += value

            pnl = value - cost
            pnl_pct = pnl / cost if cost > 0 else 0

            position_details.append({
                **pos,
                "current_price": price,
                "market_value": value,
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
            })

        return {
            "positions": position_details,
            "total_cost": total_cost,
            "total_value": total_value,
            "total_pnl": total_value - total_cost,
            "total_pnl_pct": (total_value - total_cost) / total_cost if total_cost > 0 else 0,
        }
