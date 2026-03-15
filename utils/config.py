"""
Configuration management for the prediction market trading bot.
Loads from environment variables with safe defaults for paper trading.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── API Keys ────────────────────────────────────────────────────────────
    polymarket_api_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""))
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── Market Filters ───────────────────────────────────────────────────────
    min_volume: float = 50.0          # Minimum 24h volume in USD
    max_days_to_expiry: int = 90       # Max days until market resolves
    max_spread_pct: float = 0.05      # Max bid-ask spread (5%)

    # ── Signal Thresholds ────────────────────────────────────────────────────
    min_edge: float = 0.04             # Minimum edge to generate a trade signal

    # ── Risk Parameters ──────────────────────────────────────────────────────
    bankroll: float = 10_000.0         # Starting bankroll in USD
    max_position_pct: float = 0.05     # Max % of bankroll per position
    max_concurrent_trades: int = 15    # Max open positions at once
    daily_loss_limit_pct: float = 0.15 # Stop trading if daily loss > 15%
    max_drawdown_pct: float = 0.08     # Stop trading if drawdown > 8%
    kelly_fraction: float = 0.25       # Fractional Kelly multiplier

    # ── Execution ────────────────────────────────────────────────────────────
    paper_trading: bool = True         # Always paper trade unless explicitly disabled
    slippage_bps: float = 10.0         # Simulated slippage in basis points

    # ── Database ─────────────────────────────────────────────────────────────
    db_path: str = "data/trades.db"

    # ── Dashboard ────────────────────────────────────────────────────────────
    dashboard_port: int = 8050

    # ── FastAPI ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls()

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if not (0 < self.min_edge < 1):
            raise ValueError(f"min_edge must be between 0 and 1, got {self.min_edge}")
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be (0,1], got {self.kelly_fraction}")
        if self.bankroll <= 0:
            raise ValueError(f"bankroll must be positive, got {self.bankroll}")


# Module-level singleton
config = Config.from_env()
