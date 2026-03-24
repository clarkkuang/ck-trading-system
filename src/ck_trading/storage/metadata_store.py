"""SQLite metadata store for signals, portfolio, and job tracking."""

import sqlite3
from datetime import datetime
from pathlib import Path

from ck_trading.config import settings
from ck_trading.models.portfolio import Position, Trade
from ck_trading.models.signals import Signal


class MetadataStore:
    """SQLite-based store for metadata that doesn't belong in Parquet."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or settings.metadata_db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                score REAL NOT NULL,
                rationale TEXT,
                generated_at TEXT NOT NULL,
                price_at_signal REAL,
                target_price REAL,
                stop_loss REAL,
                acted_on INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT 'US',
                shares REAL NOT NULL,
                avg_cost REAL NOT NULL,
                date_acquired TEXT NOT NULL,
                is_closed INTEGER DEFAULT 0,
                closed_at TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT 'US',
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                date TEXT NOT NULL,
                fees REAL DEFAULT 0,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL UNIQUE,
                market TEXT NOT NULL DEFAULT 'US',
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                source TEXT NOT NULL DEFAULT 'manual'
            );

            CREATE TABLE IF NOT EXISTS universe (
                ticker TEXT PRIMARY KEY,
                market TEXT NOT NULL DEFAULT 'US',
                name TEXT,
                sector TEXT,
                industry TEXT
            );

            CREATE TABLE IF NOT EXISTS job_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT DEFAULT 'running',
                result TEXT
            );
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Apply schema migrations for existing databases."""
        # Add 'source' column to watchlist if it doesn't exist
        cols = [
            row[1]
            for row in self.conn.execute("PRAGMA table_info(watchlist)").fetchall()
        ]
        if "source" not in cols:
            self.conn.execute(
                "ALTER TABLE watchlist ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'"
            )
            self.conn.commit()

    # --- Signals ---

    def save_signal(self, signal: Signal) -> int:
        cursor = self.conn.execute(
            """INSERT INTO signals
               (ticker, signal_type, strategy_name, score, rationale,
                generated_at, price_at_signal, target_price, stop_loss)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.ticker,
                signal.signal_type.value,
                signal.strategy_name,
                signal.score,
                signal.rationale,
                signal.generated_at.isoformat(),
                signal.price_at_signal,
                signal.target_price,
                signal.stop_loss,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM signals ORDER BY generated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Positions ---

    def save_position(self, position: Position) -> int:
        cursor = self.conn.execute(
            """INSERT INTO positions (ticker, market, shares, avg_cost, date_acquired)
               VALUES (?, ?, ?, ?, ?)""",
            (
                position.ticker,
                position.market.value,
                position.shares,
                position.avg_cost,
                position.date_acquired.isoformat(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def get_open_positions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM positions WHERE is_closed = 0"
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Trades ---

    def save_trade(self, trade: Trade) -> int:
        cursor = self.conn.execute(
            """INSERT INTO trades (ticker, market, action, shares, price, date, fees, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.ticker,
                trade.market.value,
                trade.action.value,
                trade.shares,
                trade.price,
                trade.date.isoformat(),
                trade.fees,
                trade.notes,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    # --- Universe ---

    def save_universe(self, tickers: list[dict]) -> None:
        """Save stock universe. Each dict has: ticker, market, name, sector, industry."""
        self.conn.executemany(
            """INSERT OR REPLACE INTO universe (ticker, market, name, sector, industry)
               VALUES (:ticker, :market, :name, :sector, :industry)""",
            tickers,
        )
        self.conn.commit()

    def add_to_universe(self, ticker: str, market: str = "us") -> None:
        """Add a single ticker to the universe (idempotent)."""
        self.conn.execute(
            """INSERT OR IGNORE INTO universe (ticker, market, name, sector, industry)
               VALUES (?, ?, '', '', '')""",
            (ticker.upper(), market.lower()),
        )
        self.conn.commit()

    def get_universe(self, market: str | None = None) -> list[dict]:
        if market:
            rows = self.conn.execute(
                "SELECT * FROM universe WHERE market = ?", (market,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM universe").fetchall()
        return [dict(r) for r in rows]

    # --- Watchlist ---

    def add_to_watchlist(
        self,
        ticker: str,
        market: str = "US",
        notes: str = "",
        source: str = "manual",
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, market, notes, source) VALUES (?, ?, ?, ?)",
            (ticker, market, notes, source),
        )
        self.conn.commit()

    def get_watchlist(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM watchlist").fetchall()
        return [dict(r) for r in rows]

    # --- Job runs ---

    def log_job_start(self, job_name: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO job_runs (job_name, started_at) VALUES (?, ?)",
            (job_name, datetime.now().isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def log_job_end(self, job_id: int, status: str = "success", result: str = "") -> None:
        self.conn.execute(
            "UPDATE job_runs SET finished_at = ?, status = ?, result = ? WHERE id = ?",
            (datetime.now().isoformat(), status, result, job_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
