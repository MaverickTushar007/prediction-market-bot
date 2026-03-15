"""
Polymarket Market Scanner
Fetches active markets, filters by liquidity/volume, ranks by opportunity.
"""
import requests
import logging
from typing import List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

def fetch_active_markets(limit: int = 100) -> List[dict]:
    """Fetch active markets from Polymarket Gamma API."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={"limit": limit, "active": "true", "closed": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch markets: {e}")
        return []


def parse_market(m: dict) -> dict:
    """Extract key fields from raw market data."""
    try:
        prices = m.get("outcomePrices", ["0.5", "0.5"])
        if isinstance(prices, str):
            import json as _json
            prices = _json.loads(prices)
        yes_price = float(prices[0]) if prices else 0.5
        no_price  = float(prices[1]) if len(prices) > 1 else round(1 - yes_price, 3)

        # Get category from events if available
        events = m.get("events", [])
        category = events[0].get("category", "") if events else m.get("category", "")

        return {
            "id":              m.get("id", ""),
            "question":        m.get("question", ""),
            "category":        category,
            "yes_price":       round(yes_price, 3),
            "no_price":        round(no_price, 3),
            "volume":          float(m.get("volumeNum", 0) or 0),
            "volume_24h":      float(m.get("volume24hr", 0) or 0),
            "liquidity":       float(m.get("liquidityNum", 0) or 0),
            "spread":          float(m.get("spread", 0) or 0),
            "last_price":      float(m.get("lastTradePrice", yes_price) or yes_price),
            "price_change_1d": float(m.get("oneDayPriceChange", 0) or 0),
            "price_change_1w": float(m.get("oneWeekPriceChange", 0) or 0),
            "end_date":        m.get("endDateIso", m.get("endDate", "")),
            "url":             f"https://polymarket.com/event/{m.get('slug','')}",
            "slug":            m.get("slug", ""),
        }
    except Exception as e:
        logger.debug(f"Parse error: {e}")
        return {}


def filter_markets(markets: List[dict],
                   min_volume: float = 1000,
                   min_liquidity: float = 500) -> List[dict]:
    """Filter markets by volume and liquidity."""
    filtered = []
    for m in markets:
        parsed = parse_market(m)
        if not parsed:
            continue
        if parsed["volume"] < min_volume:
            continue
        if parsed["liquidity"] < min_liquidity:
            continue
        # Skip markets priced at extremes (>95% or <5%) — low edge
        if parsed["yes_price"] > 0.95 or parsed["yes_price"] < 0.05:
            continue
        filtered.append(parsed)

    # Sort by volume descending
    filtered.sort(key=lambda x: x["volume"], reverse=True)
    return filtered


def scan_markets(limit: int = 50) -> List[dict]:
    """Main entry: fetch, filter, and rank markets."""
    raw     = fetch_active_markets(limit=limit)
    markets = filter_markets(raw)
    logger.info(f"Scanned {len(raw)} markets → {len(markets)} tradeable")
    return markets
