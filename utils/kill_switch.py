"""
Kill Switch — emergency stop for all trading activity.
Create a file called STOP in the project root to halt everything.
Delete it to resume.
"""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
STOP_FILE = Path("STOP")


def is_active() -> bool:
    """Returns True if kill switch is ON (trading halted)."""
    return STOP_FILE.exists()


def activate(reason: str = "Manual stop"):
    """Activate kill switch — halts all trading."""
    STOP_FILE.write_text(f"Kill switch activated: {reason}")
    logger.warning(f"🛑 KILL SWITCH ACTIVATED: {reason}")


def deactivate():
    """Deactivate kill switch — resumes trading."""
    if STOP_FILE.exists():
        STOP_FILE.unlink()
        logger.info("✅ Kill switch deactivated — trading resumed")


def check_and_raise():
    """Call before any trade — raises exception if kill switch is on."""
    if is_active():
        reason = STOP_FILE.read_text()
        raise RuntimeError(f"Kill switch active: {reason}")


def status() -> dict:
    return {
        "active": is_active(),
        "reason": STOP_FILE.read_text() if is_active() else None,
    }
