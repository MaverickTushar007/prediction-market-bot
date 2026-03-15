"""
Paper Trading P&L Tracker
Logs virtual trades for both Polymarket signals and stock signals.
Tracks running P&L, win rate, and Sharpe ratio over time.
"""
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)
TRADES_FILE = Path("data/paper_trades.json")


def _load_trades() -> list:
    if TRADES_FILE.exists():
        try:
            return json.loads(TRADES_FILE.read_text())
        except Exception:
            pass
    return []


def _save_trades(trades: list):
    TRADES_FILE.parent.mkdir(exist_ok=True)
    TRADES_FILE.write_text(json.dumps(trades, indent=2))


def log_trade(symbol: str, direction: str, entry_price: float,
              target: float, stop_loss: float, confidence: float,
              source: str = "signals", notes: str = "") -> dict:
    """Log a new paper trade."""
    trades = _load_trades()
    trade = {
        "id":          len(trades) + 1,
        "symbol":      symbol,
        "direction":   direction,
        "entry_price": round(entry_price, 6),
        "target":      round(target, 6),
        "stop_loss":   round(stop_loss, 6),
        "confidence":  confidence,
        "source":      source,  # "signals" or "polymarket"
        "notes":       notes,
        "status":      "OPEN",
        "exit_price":  None,
        "pnl_pct":     None,
        "pnl_usd":     None,
        "outcome":     None,
        "opened_at":   datetime.now(timezone.utc).isoformat(),
        "closed_at":   None,
    }
    trades.append(trade)
    _save_trades(trades)
    logger.info(f"Paper trade logged: {direction} {symbol} @ {entry_price}")
    return trade


def close_trade(trade_id: int, exit_price: float,
                position_size: float = 1000) -> Optional[dict]:
    """Close a paper trade and calculate P&L."""
    trades = _load_trades()
    for t in trades:
        if t["id"] == trade_id and t["status"] == "OPEN":
            entry = t["entry_price"]
            if t["direction"] == "BUY":
                pnl_pct = (exit_price - entry) / entry * 100
            else:
                pnl_pct = (entry - exit_price) / entry * 100

            pnl_usd = position_size * pnl_pct / 100
            outcome = "WIN" if pnl_pct > 0 else "LOSS"

            t["exit_price"] = round(exit_price, 6)
            t["pnl_pct"]    = round(pnl_pct, 2)
            t["pnl_usd"]    = round(pnl_usd, 2)
            t["outcome"]    = outcome
            t["status"]     = "CLOSED"
            t["closed_at"]  = datetime.now(timezone.utc).isoformat()

            _save_trades(trades)
            logger.info(f"Trade {trade_id} closed: {outcome} {pnl_pct:+.2f}%")
            return t
    return None


def get_pnl_stats(source: str = None) -> dict:
    """Calculate P&L statistics across all closed trades."""
    trades = _load_trades()
    closed = [t for t in trades if t["status"] == "CLOSED"]
    if source:
        closed = [t for t in closed if t["source"] == source]
    open_trades = [t for t in trades if t["status"] == "OPEN"]

    if not closed:
        return {
            "total_trades":  len(trades),
            "open_trades":   len(open_trades),
            "closed_trades": 0,
            "win_rate":      None,
            "total_pnl_pct": 0,
            "total_pnl_usd": 0,
            "avg_win":       None,
            "avg_loss":      None,
            "sharpe":        None,
            "profit_factor": None,
            "trades":        trades[-20:],
        }

    pnl_pcts  = [t["pnl_pct"] for t in closed]
    wins      = [p for p in pnl_pcts if p > 0]
    losses    = [p for p in pnl_pcts if p <= 0]
    win_rate  = len(wins) / len(closed) * 100
    total_pnl = sum(t["pnl_usd"] or 0 for t in closed)

    # Sharpe ratio
    if len(pnl_pcts) > 1:
        arr = np.array(pnl_pcts)
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 0 else 0
    else:
        sharpe = 0

    # Profit factor
    gross_win  = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999

    return {
        "total_trades":  len(trades),
        "open_trades":   len(open_trades),
        "closed_trades": len(closed),
        "win_rate":      round(win_rate, 1),
        "total_pnl_pct": round(sum(pnl_pcts), 2),
        "total_pnl_usd": round(total_pnl, 2),
        "avg_win":       round(sum(wins)/len(wins), 2) if wins else 0,
        "avg_loss":      round(sum(losses)/len(losses), 2) if losses else 0,
        "sharpe":        round(sharpe, 2),
        "profit_factor": pf,
        "best_trade":    max(pnl_pcts) if pnl_pcts else 0,
        "worst_trade":   min(pnl_pcts) if pnl_pcts else 0,
        "trades":        sorted(trades, key=lambda x: x["opened_at"], reverse=True)[:20],
    }
