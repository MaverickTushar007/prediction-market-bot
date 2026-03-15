"""
Shared helper utilities used across all modules.
"""

import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Decorators ────────────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Retry a function on failure with exponential back-off."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    wait = delay * (2 ** (attempt - 1))
                    logging.getLogger(__name__).warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {exc}. "
                        f"Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)
        return wrapper  # type: ignore
    return decorator


def timer(func: F) -> F:
    """Log the execution time of a function."""
    logger = get_logger(__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper  # type: ignore


# ── Date / Time ───────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def days_until(expiry: datetime) -> float:
    """Return number of days until *expiry* from now."""
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    delta = expiry - utcnow()
    return max(delta.total_seconds() / 86_400, 0.0)


# ── Finance Helpers ───────────────────────────────────────────────────────────

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def pct_change(old: float, new: float) -> float:
    """Safe percentage change."""
    if old == 0:
        return 0.0
    return (new - old) / abs(old)


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a sequence of period returns."""
    excess = returns - risk_free / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Return max drawdown as a positive fraction."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak == 0, 1, peak)
    return float(-np.min(dd))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score — lower is better, 0 is perfect."""
    return float(np.mean((probabilities - outcomes) ** 2))


def profit_factor(pnl_array: np.ndarray) -> float:
    """Gross profit / gross loss."""
    gains = pnl_array[pnl_array > 0].sum()
    losses = abs(pnl_array[pnl_array < 0].sum())
    return gains / losses if losses > 0 else float("inf")
