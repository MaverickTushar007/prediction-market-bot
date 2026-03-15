"""
Trade Executor — orchestrates risk checks → sizing → paper execution → logging.
"""

from __future__ import annotations

from typing import Optional

from data.trade_logger import TradeLogger
from execution.paper_trader import PaperTrader, Trade
from prediction.ensemble_model import PredictionSignal
from risk.kelly import calculate_bet_size
from risk.risk_checks import RiskManager
from utils.config import config
from utils.helpers import get_logger

logger = get_logger(__name__)


class TradeExecutor:
    """
    High-level trade orchestrator.

    Flow:
      1. Receive a PredictionSignal
      2. Run Kelly sizing
      3. Run risk checks
      4. Execute via PaperTrader
      5. Log to SQLite
    """

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        paper_trader: Optional[PaperTrader] = None,
        trade_logger: Optional[TradeLogger] = None,
    ):
        self.risk = risk_manager or RiskManager()
        self.trader = paper_trader or PaperTrader()
        self.logger_db = trade_logger or TradeLogger()

    def execute(self, signal: PredictionSignal, question: str = "") -> Optional[Trade]:
        """
        Attempt to execute a trade from a prediction signal.
        Returns the Trade object if executed, None otherwise.
        """
        if not signal.has_signal:
            logger.debug(f"[{signal.market_id}] PASS — edge insufficient ({signal.edge:+.4f}).")
            return None

        # ── Kelly sizing ───────────────────────────────────────────────────
        sizing = calculate_bet_size(
            market_id=signal.market_id,
            direction=signal.direction,
            market_price=signal.market_price,
            model_probability=signal.model_probability,
            bankroll=self.risk.current_bankroll,
        )

        # ── Risk checks ────────────────────────────────────────────────────
        risk_result = self.risk.check_trade(sizing.bet_size_usd, signal.market_id)
        if not risk_result:
            logger.warning(f"[{signal.market_id}] Risk check FAILED: {risk_result.reason}")
            return None

        # ── Paper execution ────────────────────────────────────────────────
        trade = self.trader.open_trade(
            market_id=signal.market_id,
            question=question or signal.question,
            direction=signal.direction,
            size_usd=sizing.bet_size_usd,
            market_price=signal.market_price,
            model_probability=signal.model_probability,
            edge=signal.edge,
        )

        if trade is None:
            return None

        # ── Update risk state ──────────────────────────────────────────────
        self.risk.record_open(trade.to_dict())

        # ── Persist to SQLite ──────────────────────────────────────────────
        self.logger_db.log_trade(trade)

        return trade

    def resolve(self, market_id: str, outcome: bool) -> Optional[Trade]:
        """Close and settle an open position."""
        trade = self.trader.close_trade(market_id, outcome)
        if trade is None:
            return None
        self.risk.record_close(market_id, trade.pnl or 0.0)
        self.logger_db.update_trade(trade)
        return trade
