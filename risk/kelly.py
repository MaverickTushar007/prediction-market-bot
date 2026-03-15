"""
Kelly Criterion — optimal position sizing.

Formula:
    f* = (p * b - q) / b

Where:
    p = probability of winning (model estimate)
    q = 1 - p (probability of losing)
    b = net odds (payout ratio)

For binary prediction markets where YES pays (1/price - 1):
    b = (1 - price) / price

We use fractional Kelly (default 0.25×) to reduce variance.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.config import config
from utils.helpers import clamp, get_logger

logger = get_logger(__name__)


@dataclass
class KellySizing:
    """Result of Kelly calculation for a single trade."""
    market_id: str
    direction: str           # "YES" | "NO"
    p_win: float             # Model probability
    p_lose: float
    b: float                 # Net odds
    full_kelly: float        # Full Kelly fraction
    fractional_kelly: float  # After applying multiplier
    bet_size_usd: float      # Dollar amount to wager
    bankroll: float
    max_position_usd: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def kelly_fraction(p: float, b: float) -> float:
    """
    Compute the full Kelly fraction f* = (p*b - q) / b.

    Returns 0 if the bet has no positive expected value.
    """
    q = 1 - p
    edge = p * b - q
    if edge <= 0 or b <= 0:
        return 0.0
    return clamp(edge / b, 0.0, 1.0)


def calculate_bet_size(
    market_id: str,
    direction: str,
    market_price: float,
    model_probability: float,
    bankroll: float = config.bankroll,
    kelly_multiplier: float = config.kelly_fraction,
    max_position_pct: float = config.max_position_pct,
) -> KellySizing:
    """
    Calculate the optimal bet size for a market.

    For YES bets:  price = cost per contract, payout = $1
    For NO bets:   treat (1 - price) as the cost, same $1 payout

    Returns a KellySizing dataclass with all intermediate values.
    """
    if direction == "YES":
        cost = market_price
        p_win = model_probability
    else:
        cost = 1.0 - market_price
        p_win = 1.0 - model_probability

    p_lose = 1.0 - p_win

    # Net odds: if you bet $cost and win, you receive $1 → net = (1-cost)/cost
    if cost <= 0 or cost >= 1:
        b = 0.0
    else:
        b = (1.0 - cost) / cost

    fk = kelly_fraction(p_win, b)
    frac_kelly = fk * kelly_multiplier

    # Dollar amount — capped at max_position_pct of bankroll
    max_position_usd = bankroll * max_position_pct
    bet_size_usd = min(bankroll * frac_kelly, max_position_usd)
    bet_size_usd = max(0.0, bet_size_usd)

    logger.debug(
        f"Kelly [{market_id}] dir={direction} p={p_win:.3f} b={b:.2f} "
        f"fk={fk:.4f} frac={frac_kelly:.4f} bet=${bet_size_usd:.2f}"
    )

    return KellySizing(
        market_id=market_id,
        direction=direction,
        p_win=round(p_win, 4),
        p_lose=round(p_lose, 4),
        b=round(b, 4),
        full_kelly=round(fk, 6),
        fractional_kelly=round(frac_kelly, 6),
        bet_size_usd=round(bet_size_usd, 2),
        bankroll=bankroll,
        max_position_usd=round(max_position_usd, 2),
    )
