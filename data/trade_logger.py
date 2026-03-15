"""
Trade Logger — persists trades and pipeline runs to SQLite.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from data.database import get_connection, init_db
from execution.paper_trader import Trade
from utils.config import config
from utils.helpers import get_logger, utcnow

logger = get_logger(__name__)


class TradeLogger:
    """Read/write interface to the trades SQLite table."""

    def __init__(self, db_path: str = config.db_path):
        self.db_path = db_path
        init_db(db_path)

    def _conn(self):
        return get_connection(self.db_path)

    def log_trade(self, trade: Trade) -> None:
        """Insert a new trade row."""
        d = trade.to_dict()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trades
                   (trade_id, market_id, question, direction, entry_price,
                    size_usd, contracts, model_prob, market_price, edge,
                    status, opened_at, closed_at, exit_price, pnl, outcome)
                   VALUES (:trade_id,:market_id,:question,:direction,:entry_price,
                           :size_usd,:contracts,:model_probability,:market_price,:edge,
                           :status,:opened_at,:closed_at,:exit_price,:pnl,:outcome)""",
                {
                    **d,
                    "model_probability": d.get("model_probability"),
                    "outcome": int(d["outcome"]) if d.get("outcome") is not None else None,
                },
            )
        logger.debug(f"Logged trade {trade.trade_id} to DB.")

    def update_trade(self, trade: Trade) -> None:
        """Update exit fields when a trade is closed."""
        self.log_trade(trade)

    def log_research(
        self,
        market_id: str,
        question: str,
        sentiment_label: str,
        sentiment_score: float,
        model_prob: float,
        market_price: float,
        edge: float,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO market_research
                   (market_id, question, sentiment_label, sentiment_score,
                    model_prob, market_price, edge)
                   VALUES (?,?,?,?,?,?,?)""",
                (market_id, question, sentiment_label, sentiment_score,
                 model_prob, market_price, edge),
            )

    def log_pipeline_run(
        self,
        started_at: str,
        finished_at: str,
        markets_scanned: int,
        signals_found: int,
        trades_placed: int,
        notes: str = "",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO pipeline_runs
                   (started_at, finished_at, markets_scanned, signals_found, trades_placed, notes)
                   VALUES (?,?,?,?,?,?)""",
                (started_at, finished_at, markets_scanned, signals_found, trades_placed, notes),
            )

    def get_trades(self, status: Optional[str] = None) -> pd.DataFrame:
        """Return trades as a DataFrame, optionally filtered by status."""
        query = "SELECT * FROM trades"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY opened_at DESC"
        with self._conn() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_pipeline_runs(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC", conn
            )

    def get_research(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM market_research ORDER BY researched_at DESC", conn
            )
