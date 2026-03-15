"""
Database — SQLite schema and connection management.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from utils.config import config
from utils.helpers import get_logger

logger = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    market_id       TEXT NOT NULL,
    question        TEXT,
    direction       TEXT,
    entry_price     REAL,
    size_usd        REAL,
    contracts       REAL,
    model_prob      REAL,
    market_price    REAL,
    edge            REAL,
    status          TEXT,
    opened_at       TEXT,
    closed_at       TEXT,
    exit_price      REAL,
    pnl             REAL,
    outcome         INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT,
    finished_at     TEXT,
    markets_scanned INTEGER,
    signals_found   INTEGER,
    trades_placed   INTEGER,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS market_research (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id       TEXT,
    question        TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    model_prob      REAL,
    market_price    REAL,
    edge            REAL,
    researched_at   TEXT DEFAULT (datetime('now'))
);
"""


def get_connection(db_path: str = config.db_path) -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB file if needed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = config.db_path) -> None:
    """Create tables if they do not already exist."""
    conn = get_connection(db_path)
    conn.executescript(_DDL)
    conn.commit()
    conn.close()
    logger.info(f"Database initialised at {db_path}")
