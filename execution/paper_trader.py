"""
Paper Trader — simulates order execution without real money.

Simulates:
  • Limit orders with fill probability
  • Slippage
  • Position tracking
  • Outcome resolution (random in mock mode)
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from utils.config import config
from utils.helpers import get_logger, utcnow

logger = get_logger(__name__)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id: str
    market_id: str
    question: str
    direction: str           # "YES" | "NO"
    entry_price: float       # After slippage
    size_usd: float
    contracts: float         # size_usd / entry_price
    model_probability: float
    market_price: float
    edge: float
    status: str              # "open" | "closed" | "cancelled"
    opened_at: datetime
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[bool] = None  # True = YES resolved

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["opened_at"] = self.opened_at.isoformat()
        if self.closed_at:
            d["closed_at"] = self.closed_at.isoformat()
        return d

    @property
    def unrealised_pnl(self) -> float:
        """Approximate unrealised PnL (not available in pure paper mode)."""
        return 0.0

    @property
    def return_pct(self) -> Optional[float]:
        if self.pnl is None or self.size_usd == 0:
            return None
        return self.pnl / self.size_usd


# ── Paper trader ──────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Simulates trade execution and outcome resolution.
    All operations are in-memory; the TradeLogger handles persistence.
    """

    def __init__(self, slippage_bps: float = config.slippage_bps):
        self.slippage_bps = slippage_bps
        self._open: Dict[str, Trade] = {}   # market_id → Trade
        self._closed: List[Trade] = []

    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply simulated market impact. YES buys push price up slightly."""
        slip = self.slippage_bps / 10_000
        if direction == "YES":
            return min(price * (1 + slip), 0.99)
        else:
            return min(price * (1 + slip), 0.99)

    def _fill_probability(self, price: float) -> float:
        """
        Simulate partial fill probability for limit orders.
        Very illiquid prices (<5% or >95%) have lower fill rates.
        """
        if price < 0.05 or price > 0.95:
            return 0.70
        return 0.95

    def open_trade(
        self,
        market_id: str,
        question: str,
        direction: str,
        size_usd: float,
        market_price: float,
        model_probability: float,
        edge: float,
    ) -> Optional[Trade]:
        """
        Attempt to open a paper trade.
        Returns the Trade if filled, None if simulated fill failure.
        """
        if market_id in self._open:
            logger.warning(f"Already have open position in {market_id} — skipping.")
            return None

        fill_p = self._fill_probability(market_price)
        if random.random() > fill_p:
            logger.info(f"Trade not filled (simulated): {market_id}")
            return None

        entry_price = self._apply_slippage(market_price, direction)
        contracts = size_usd / entry_price

        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            market_id=market_id,
            question=question,
            direction=direction,
            entry_price=round(entry_price, 4),
            size_usd=round(size_usd, 2),
            contracts=round(contracts, 4),
            model_probability=model_probability,
            market_price=market_price,
            edge=edge,
            status="open",
            opened_at=utcnow(),
        )

        self._open[market_id] = trade
        logger.info(
            f"TRADE OPEN  [{trade.trade_id}] {market_id} {direction} "
            f"${size_usd:.2f} @ {entry_price:.4f} | edge={edge:+.4f}"
        )
        return trade

    def close_trade(
        self,
        market_id: str,
        outcome: bool,
        exit_price: Optional[float] = None,
    ) -> Optional[Trade]:
        """
        Resolve a trade. *outcome*=True means YES resolved.
        """
        trade = self._open.pop(market_id, None)
        if trade is None:
            logger.warning(f"No open trade found for {market_id}")
            return None

        won = (trade.direction == "YES" and outcome) or (trade.direction == "NO" and not outcome)

        if exit_price is None:
            exit_price = 1.0 if won else 0.0

        # PnL = contracts × (exit_price - entry_price) × entry_price
        # Simplified: if win, receive $1 per contract; if lose, receive $0
        pnl = trade.contracts * exit_price - trade.size_usd
        pnl = round(pnl, 2)

        trade.closed_at = utcnow()
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.outcome = outcome
        trade.status = "closed"

        self._closed.append(trade)
        logger.info(
            f"TRADE CLOSE [{trade.trade_id}] {market_id} outcome={'YES' if outcome else 'NO'} "
            f"PnL=${pnl:+.2f} ({'+' if pnl>=0 else ''}{trade.return_pct*100:.1f}%)"
        )
        return trade

    def simulate_random_resolution(self, market_id: str, true_prob: float) -> Optional[Trade]:
        """Randomly resolve an open trade using the model's probability estimate."""
        outcome = random.random() < true_prob
        return self.close_trade(market_id, outcome)

    @property
    def open_trades(self) -> List[Trade]:
        return list(self._open.values())

    @property
    def closed_trades(self) -> List[Trade]:
        return self._closed.copy()

    def performance_stats(self) -> dict:
        """Calculate performance metrics from closed trades."""
        if not self._closed:
            return {"message": "No closed trades yet."}

        pnls = np.array([t.pnl for t in self._closed if t.pnl is not None])
        outcomes = np.array([int(t.pnl > 0) for t in self._closed if t.pnl is not None])
        probs = np.array([t.model_probability for t in self._closed if t.pnl is not None])
        actuals = np.array([int(t.outcome) for t in self._closed if t.outcome is not None])

        win_rate = float(np.mean(outcomes)) if len(outcomes) else 0.0
        gross_profit = float(pnls[pnls > 0].sum()) if any(pnls > 0) else 0.0
        gross_loss = float(abs(pnls[pnls < 0].sum())) if any(pnls < 0) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Brier score
        brier = float(np.mean((probs - actuals) ** 2)) if len(probs) == len(actuals) else None

        from utils.helpers import sharpe_ratio as _sharpe
        sr = _sharpe(pnls / 100) if len(pnls) > 1 else 0.0

        return {
            "total_trades": len(self._closed),
            "open_trades": len(self._open),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(float(pnls.sum()), 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sr, 4),
            "brier_score": round(brier, 4) if brier is not None else None,
        }
