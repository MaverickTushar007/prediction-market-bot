"""
Kalshi Market Scanner
Uses Kalshi demo API for market discovery.
Real API requires account + authentication.
"""
import requests
import logging
from typing import List

logger = logging.getLogger(__name__)
DEMO_API = "https://demo-api.kalshi.co/trade-api/v2"
LIVE_API = "https://trading-api.kalshi.com/trade-api/v2"


def fetch_kalshi_markets(limit: int = 50, use_demo: bool = True) -> List[dict]:
    """Fetch open markets from Kalshi."""
    base = DEMO_API if use_demo else LIVE_API
    try:
        resp = requests.get(
            f"{base}/markets",
            params={"limit": limit, "status": "open"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("markets", [])
    except Exception as e:
        logger.warning(f"Kalshi fetch failed: {e}")
        return []


def parse_kalshi_market(m: dict) -> dict:
    """Parse Kalshi market into standard format."""
    try:
        yes_ask = float(m.get("yes_ask", 50)) / 100  # Kalshi uses cents
        no_ask  = float(m.get("no_ask", 50)) / 100
        yes_bid = float(m.get("yes_bid", 50)) / 100

        return {
            "id":           m.get("ticker", ""),
            "question":     m.get("title", ""),
            "category":     m.get("category", ""),
            "yes_price":    round(yes_ask, 3),
            "no_price":     round(no_ask, 3),
            "yes_bid":      round(yes_bid, 3),
            "volume":       float(m.get("volume", 0) or 0),
            "open_interest":float(m.get("open_interest", 0) or 0),
            "end_date":     m.get("close_time", ""),
            "url":          f"https://kalshi.com/markets/{m.get('ticker','')}",
            "source":       "kalshi",
        }
    except Exception as e:
        logger.debug(f"Kalshi parse error: {e}")
        return {}


def scan_kalshi(limit: int = 30) -> List[dict]:
    """Scan Kalshi for tradeable markets."""
    raw     = fetch_kalshi_markets(limit=limit)
    markets = []
    for m in raw:
        parsed = parse_kalshi_market(m)
        if not parsed:
            continue
        # Filter: skip extremes
        if parsed["yes_price"] > 0.95 or parsed["yes_price"] < 0.05:
            continue
        markets.append(parsed)

    logger.info(f"Kalshi: {len(raw)} raw → {len(markets)} tradeable")
    return markets
