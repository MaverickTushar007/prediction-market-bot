"""
Risk Checks — Step 4 of the pipeline.

Hard stops and circuit breakers that must pass before any trade executes.
Rules:
  • Max position = 5% of bankroll
  • Max concurrent trades = 15
  • Daily loss limit = 15%
  • Max drawdown = 8%
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List

from utils.config import config
from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class RiskCheckResult:
    passed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.passed


class RiskManager:
    """
    Stateful risk manager that tracks open positions and PnL history
    to enforce circuit breakers.
    """

    def __init__(
        self,
        bankroll: float = config.bankroll,
        max_position_pct: float = config.max_position_pct,
        max_concurrent: int = config.max_concurrent_trades,
        daily_loss_limit_pct: float = config.daily_loss_limit_pct,
        max_drawdown_pct: float = config.max_drawdown_pct,
    ):
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.peak_bankroll = bankroll
        self.max_position_pct = max_position_pct
        self.max_concurrent = max_concurrent
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct

        self._open_positions: List[dict] = []
        self._daily_start_bankroll: float = bankroll
        self._daily_start_date: date = datetime.now(timezone.utc).date()
        self._halted: bool = False

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def current_drawdown(self) -> float:
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll

    @property
    def daily_pnl_pct(self) -> float:
        if self._daily_start_bankroll <= 0:
            return 0.0
        return (self.current_bankroll - self._daily_start_bankroll) / self._daily_start_bankroll

    def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._daily_start_date:
            self._daily_start_bankroll = self.current_bankroll
            self._daily_start_date = today

    def check_trade(self, bet_size_usd: float, market_id: str = "") -> RiskCheckResult:
        """
        Run all pre-trade risk checks.
        Returns RiskCheckResult(passed=True) if all checks pass.
        """
        self._reset_daily_if_needed()

        if self._halted:
            return RiskCheckResult(False, "Trading halted by risk manager — manual review required.")

        # 1. Max position size
        max_bet = self.current_bankroll * self.max_position_pct
        if bet_size_usd > max_bet:
            return RiskCheckResult(
                False,
                f"Bet ${bet_size_usd:.2f} exceeds max position ${max_bet:.2f} "
                f"({self.max_position_pct*100:.1f}% of bankroll).",
            )

        # 2. Concurrent trades
        if self.open_position_count >= self.max_concurrent:
            return RiskCheckResult(
                False,
                f"Max concurrent trades reached ({self.open_position_count}/{self.max_concurrent}).",
            )

        # 3. Daily loss limit
        if self.daily_pnl_pct <= -self.daily_loss_limit_pct:
            self._halted = True
            return RiskCheckResult(
                False,
                f"Daily loss limit hit: {self.daily_pnl_pct*100:.1f}% "
                f"(limit={-self.daily_loss_limit_pct*100:.1f}%).",
            )

        # 4. Max drawdown
        if self.current_drawdown >= self.max_drawdown_pct:
            self._halted = True
            return RiskCheckResult(
                False,
                f"Max drawdown breached: {self.current_drawdown*100:.1f}% "
                f"(limit={self.max_drawdown_pct*100:.1f}%).",
            )

        # 5. Minimum bet size
        if bet_size_usd < 1.0:
            return RiskCheckResult(
                False,
                f"Bet size ${bet_size_usd:.2f} too small (minimum $1.00).",
            )

        return RiskCheckResult(True, "All risk checks passed.")

    def record_open(self, trade: dict) -> None:
        """Register a new open position."""
        self._open_positions.append(trade)

    def record_close(self, market_id: str, pnl: float) -> None:
        """Close a position and update bankroll."""
        self._open_positions = [
            p for p in self._open_positions if p.get("market_id") != market_id
        ]
        self.current_bankroll = round(self.current_bankroll + pnl, 2)
        self.peak_bankroll = max(self.peak_bankroll, self.current_bankroll)
        logger.info(
            f"Position closed [{market_id}] PnL=${pnl:+.2f} | "
            f"Bankroll=${self.current_bankroll:.2f} | "
            f"Drawdown={self.current_drawdown*100:.1f}%"
        )

    def summary(self) -> dict:
        return {
            "bankroll": self.current_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "current_drawdown_pct": round(self.current_drawdown * 100, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct * 100, 2),
            "open_positions": self.open_position_count,
            "halted": self._halted,
        }
