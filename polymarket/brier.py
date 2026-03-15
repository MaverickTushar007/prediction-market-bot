"""
Brier Score Calibration Tracker
Logs every prediction, checks outcomes when markets resolve,
calculates calibration score over time.

Brier Score = mean((predicted_prob - actual_outcome)^2)
- Perfect calibration: 0.0
- Random guessing: 0.25
- Target: < 0.25
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)
BRIER_FILE = Path("data/brier_log.json")


def _load_log() -> list:
    if BRIER_FILE.exists():
        try:
            return json.loads(BRIER_FILE.read_text())
        except Exception:
            pass
    return []


def _save_log(log: list):
    BRIER_FILE.parent.mkdir(exist_ok=True)
    BRIER_FILE.write_text(json.dumps(log, indent=2))


def log_prediction(market_id: str, question: str,
                   predicted_prob: float, market_price: float,
                   recommendation: str) -> dict:
    """Log a new prediction for future scoring."""
    log = _load_log()
    entry = {
        "market_id":      market_id,
        "question":       question[:100],
        "predicted_prob": round(predicted_prob, 3),
        "market_price":   round(market_price, 3),
        "recommendation": recommendation,
        "logged_at":      datetime.now(timezone.utc).isoformat(),
        "resolved":       False,
        "outcome":        None,
        "brier_score":    None,
    }
    # Avoid duplicates
    if not any(e["market_id"] == market_id for e in log):
        log.append(entry)
        _save_log(log)
        logger.info(f"Logged prediction: {question[:40]} @ {predicted_prob:.0%}")
    return entry


def resolve_prediction(market_id: str, outcome: int) -> Optional[dict]:
    """
    Resolve a prediction with actual outcome.
    outcome: 1 = YES happened, 0 = NO happened
    """
    log = _load_log()
    for entry in log:
        if entry["market_id"] == market_id and not entry["resolved"]:
            brier = (entry["predicted_prob"] - outcome) ** 2
            entry["resolved"]    = True
            entry["outcome"]     = outcome
            entry["brier_score"] = round(brier, 4)
            entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
            _save_log(log)
            logger.info(f"Resolved {market_id}: outcome={outcome} brier={brier:.4f}")
            return entry
    return None


def get_calibration_stats() -> dict:
    """Calculate overall calibration statistics."""
    log = _load_log()
    resolved = [e for e in log if e["resolved"] and e["brier_score"] is not None]

    if not resolved:
        return {
            "total_predictions": len(log),
            "resolved":          0,
            "pending":           len(log),
            "brier_score":       None,
            "calibration":       "Insufficient data",
            "win_rate":          None,
            "predictions":       log,
        }

    brier_scores = [e["brier_score"] for e in resolved]
    avg_brier    = sum(brier_scores) / len(brier_scores)

    # Win rate — did we recommend the right side?
    correct = 0
    for e in resolved:
        if e["recommendation"] == "BUY_YES" and e["outcome"] == 1:
            correct += 1
        elif e["recommendation"] == "BUY_NO" and e["outcome"] == 0:
            correct += 1
        elif e["recommendation"] == "PASS":
            correct += 1  # passing a bad trade is also correct

    win_rate = correct / len(resolved) if resolved else 0

    # Calibration rating
    if avg_brier < 0.10:
        calibration = "Excellent"
    elif avg_brier < 0.20:
        calibration = "Good"
    elif avg_brier < 0.25:
        calibration = "Fair"
    else:
        calibration = "Poor — needs improvement"

    # Bucket analysis (reliability diagram data)
    buckets = {}
    for e in resolved:
        bucket = round(e["predicted_prob"] * 10) / 10  # 0.1 buckets
        if bucket not in buckets:
            buckets[bucket] = {"predicted": [], "actual": []}
        buckets[bucket]["predicted"].append(e["predicted_prob"])
        buckets[bucket]["actual"].append(e["outcome"])

    reliability = [
        {
            "predicted_avg": round(sum(v["predicted"])/len(v["predicted"]), 2),
            "actual_rate":   round(sum(v["actual"])/len(v["actual"]), 2),
            "count":         len(v["predicted"]),
        }
        for v in buckets.values()
    ]

    return {
        "total_predictions": len(log),
        "resolved":          len(resolved),
        "pending":           len(log) - len(resolved),
        "brier_score":       round(avg_brier, 4),
        "calibration":       calibration,
        "win_rate":          round(win_rate * 100, 1),
        "reliability":       sorted(reliability, key=lambda x: x["predicted_avg"]),
        "predictions":       log[-20:],  # last 20
    }
