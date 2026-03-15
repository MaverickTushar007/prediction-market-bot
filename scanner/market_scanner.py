"""
Market Scanner — Step 1 of the pipeline.

Connects to Polymarket and Kalshi APIs (or uses mock data when keys are absent),
filters markets by liquidity/spread/expiry, and returns a ranked list of
opportunities as a DataFrame.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from utils.config import config
from utils.helpers import days_until, get_logger, retry, utcnow

logger = get_logger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Market:
    """Normalised representation of a prediction market."""
    market_id: str
    source: str              # "polymarket" | "kalshi" | "mock"
    question: str
    category: str
    yes_price: float         # Current YES price (0-1)
    no_price: float          # Current NO price (0-1)
    volume_24h: float        # USD volume in last 24 h
    open_interest: float     # USD outstanding
    expiry: datetime
    spread: float            # (ask - bid) as fraction
    url: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def days_to_expiry(self) -> float:
        return days_until(self.expiry)

    @property
    def mid_price(self) -> float:
        return (self.yes_price + (1 - self.no_price)) / 2

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["days_to_expiry"] = self.days_to_expiry
        d["mid_price"] = self.mid_price
        d["expiry"] = self.expiry.isoformat()
        return d


# ── Mock data generator ───────────────────────────────────────────────────────

_MOCK_TEMPLATES = [
    ("Will the Fed cut rates in {month} {year}?", "economics"),
    ("Will {team} win the {league} championship?", "sports"),
    ("Will {country} hold elections before {month} {year}?", "politics"),
    ("Will {coin} reach ${price}k by {month} {year}?", "crypto"),
    ("Will {company} announce layoffs in Q{q} {year}?", "business"),
    ("Will AI replace {job} roles by {year}?", "technology"),
    ("Will {country} GDP growth exceed {pct}% in {year}?", "economics"),
]

_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]

_FILLERS = {
    "month": _MONTHS,
    "year": ["2025", "2026"],
    "team": ["Lakers", "Chiefs", "Real Madrid", "Mumbai Indians"],
    "league": ["NBA", "NFL", "La Liga", "IPL"],
    "country": ["USA", "India", "Germany", "Brazil", "Japan"],
    "coin": ["Bitcoin", "Ethereum", "Solana"],
    "price": ["80", "100", "5", "200"],
    "company": ["Meta", "Google", "Tesla", "Amazon"],
    "q": ["1", "2", "3", "4"],
    "job": ["software engineers", "accountants", "truck drivers", "lawyers"],
    "pct": ["2", "3", "4", "5"],
    "job": ["analysts", "developers", "designers"],
}


def _render_template(template: str, category: str) -> tuple[str, str]:
    """Fill a question template with random values."""
    import re
    placeholders = re.findall(r"\{(\w+)\}", template)
    for ph in placeholders:
        choices = _FILLERS.get(ph, [ph])
        template = template.replace("{" + ph + "}", random.choice(choices), 1)
    return template, category


def generate_mock_markets(n: int = 40, seed: int = 42) -> List[Market]:
    """Generate *n* realistic-looking mock markets for development/testing."""
    random.seed(seed)
    markets: List[Market] = []
    now = utcnow()

    for i in range(n):
        tmpl, cat = random.choice(_MOCK_TEMPLATES)
        question, category = _render_template(tmpl, cat)

        # Bimodal distribution — most markets cluster near 0.2 or 0.8
        yes_price = round(random.choice([
            random.gauss(0.25, 0.12),
            random.gauss(0.75, 0.12),
            random.uniform(0.3, 0.7),
        ]), 3)
        yes_price = max(0.02, min(0.98, yes_price))
        spread = round(random.uniform(0.005, 0.06), 4)
        no_price = round(1 - yes_price - spread, 3)
        no_price = max(0.01, no_price)

        expiry_days = random.randint(1, 60)
        expiry = now + timedelta(days=expiry_days)
        volume = round(random.lognormvariate(7, 1.5), 2)   # log-normal, median ~$1k
        oi = round(volume * random.uniform(0.5, 3), 2)

        markets.append(Market(
            market_id=f"mock_{i:04d}",
            source="mock",
            question=question,
            category=category,
            yes_price=yes_price,
            no_price=no_price,
            volume_24h=volume,
            open_interest=oi,
            expiry=expiry,
            spread=spread,
            url=f"https://mock.markets/{i:04d}",
            tags=[category],
        ))

    return markets


# ── Real API clients (stubs with fallback) ────────────────────────────────────

class PolymarketClient:
    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    @retry(max_attempts=2)
    def fetch_markets(self) -> List[dict]:
        """Fetch active markets from Polymarket CLOB API."""
        if False:  # API key not required for public data
            return []
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(f"{self.BASE_URL}/markets", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])


class KalshiClient:
    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    @retry(max_attempts=2)
    def fetch_markets(self) -> List[dict]:
        """Fetch active markets from Kalshi API."""
        if not self.api_key:
            logger.info("No Kalshi API key — skipping real fetch.")
            return []
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(
            f"{self.BASE_URL}/markets",
            headers=headers,
            params={"status": "open", "limit": 100},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("markets", [])


# ── Scanner ────────────────────────────────────────────────────────────────────


class GammaPolymarketClient:
    """Fetches LIVE markets from Polymarket Gamma API (no key required)."""

    BASE_URL = "https://gamma-api.polymarket.com/markets"

    def fetch_markets(self, limit: int = 100) -> list:
        import requests
        from datetime import datetime, timezone
        try:
            params = {"active": "true", "closed": "false", "limit": limit}
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            markets = []
            for m in data:
                try:
                    import json as _json
                    _prices = _json.loads(m.get("outcomePrices","[0.5,0.5]"))
                    yes_price = float(_prices[0])
                    no_price  = 1.0 - yes_price
                    spread    = abs(yes_price + no_price - 1.0) + 0.01
                    end_raw   = m.get("endDateIso") or m.get("endDate","")[:10]
                    end_dt    = datetime.strptime(end_raw[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    days_left = (end_dt - datetime.now(timezone.utc)).days
                    markets.append(Market(
                        market_id   = f"poly_{m['id']}",
                        question    = m.get("question",""),
                        yes_price   = yes_price,
                        no_price    = no_price,
                        volume_24h  = float(m.get("volume24hr", 0)),
                        expiry      = end_dt,
                        spread      = spread,
                        source      = "polymarket_live",
                        category    = "crypto",
                        open_interest = float(m.get("liquidityNum", 0)),
                    ))
                except Exception:
                    continue
            logger.info(f"GammaPolymarket: fetched {len(markets)} live markets.")
            return markets
        except Exception as e:
            logger.warning(f"GammaPolymarket fetch failed: {e}")
            return []

class MarketScanner:
    """
    Scans prediction markets, applies liquidity/spread/expiry filters,
    and returns a ranked DataFrame of candidate markets.
    """

    def __init__(
        self,
        min_volume: float = config.min_volume,
        max_days: int = config.max_days_to_expiry,
        max_spread: float = config.max_spread_pct,
    ):
        self.min_volume = min_volume
        self.max_days = max_days
        self.max_spread = max_spread
        self._poly = PolymarketClient(config.polymarket_api_key)
        self._kalshi = KalshiClient(config.kalshi_api_key)

    def _fetch_all(self) -> List[Market]:
        """Fetch from all sources, fall back to mocks."""
        markets: List[Market] = []

        # Live markets via Gamma API (no key needed)
        try:
            gamma = GammaPolymarketClient().fetch_markets(100)
            markets.extend(gamma)
        except Exception as e:
            logger.warning(f"Gamma fetch failed: {e}")

        try:
            raw = self._poly.fetch_markets()
            markets.extend(self._parse_polymarket(raw))
        except Exception as e:
            logger.warning(f"Polymarket fetch failed: {e}")

        try:
            raw = self._kalshi.fetch_markets()
            markets.extend(self._parse_kalshi(raw))
        except Exception as e:
            logger.warning(f"Kalshi fetch failed: {e}")

        if not markets:
            logger.info("No live data — using mock markets.")
            markets = generate_mock_markets(n=50)

        # Also fall back if we got data but it's all expired
        active = [m for m in markets if m.days_to_expiry > 0]
        if not active:
            logger.info("All fetched markets are expired — using mock markets.")
            markets = generate_mock_markets(n=50)

        return markets

    def _parse_polymarket(self, raw: List[dict]) -> List[Market]:
        out = []
        for m in raw:
            try:
                tokens = m.get("tokens", [])
                yes_tok = next((t for t in tokens if t.get("outcome") == "Yes"), {})
                no_tok = next((t for t in tokens if t.get("outcome") == "No"), {})
                yes_price = float(yes_tok.get("price", 0.5))
                no_price = float(no_tok.get("price", 0.5))
                expiry_ts = m.get("end_date_iso") or m.get("endDate")
                expiry = datetime.fromisoformat(str(expiry_ts).replace("Z", "+00:00"))
                out.append(Market(
                    market_id=str(m["condition_id"]),
                    source="polymarket",
                    question=m.get("question", ""),
                    category=m.get("category", "other"),
                    yes_price=yes_price,
                    no_price=no_price,
                    volume_24h=float(m.get("volume24hr", 0)),
                    open_interest=float(m.get("liquidityNum", 0)),
                    expiry=expiry,
                    spread=abs(yes_price + no_price - 1),
                    url=f"https://polymarket.com/event/{m.get('slug', '')}",
                    tags=[m.get("category", "")],
                ))
            except Exception:
                continue
        return out

    def _parse_kalshi(self, raw: List[dict]) -> List[Market]:
        out = []
        for m in raw:
            try:
                yes_price = float(m.get("yes_ask", 50)) / 100
                no_price = float(m.get("no_ask", 50)) / 100
                expiry = datetime.fromisoformat(
                    m["close_time"].replace("Z", "+00:00")
                )
                out.append(Market(
                    market_id=str(m["ticker"]),
                    source="kalshi",
                    question=m.get("title", ""),
                    category=m.get("category", "other"),
                    yes_price=yes_price,
                    no_price=no_price,
                    volume_24h=float(m.get("volume", 0)),
                    open_interest=float(m.get("open_interest", 0)),
                    expiry=expiry,
                    spread=abs(yes_price + no_price - 1),
                    url=f"https://kalshi.com/markets/{m.get('ticker', '')}",
                    tags=[m.get("category", "")],
                ))
            except Exception:
                continue
        return out

    def _filter(self, markets: List[Market]) -> List[Market]:
        """Apply liquidity / expiry / spread filters."""
        filtered = [
            m for m in markets
            if m.volume_24h >= self.min_volume
            and 0 < m.days_to_expiry <= self.max_days
            and m.spread <= self.max_spread
        ]
        logger.info(
            f"Filter: {len(markets)} total → {len(filtered)} pass "
            f"(vol≥{self.min_volume}, days≤{self.max_days}, spread≤{self.max_spread*100:.1f}%)"
        )
        return filtered

    def _rank(self, markets: List[Market]) -> pd.DataFrame:
        """
        Rank markets by a composite liquidity/uncertainty score.
        Score = log(volume) * (1 - |mid - 0.5| * 2)
        High volume + high uncertainty = most interesting.
        """
        import math
        rows = []
        for m in markets:
            uncertainty = 1 - abs(m.mid_price - 0.5) * 2   # 1 at 50%, 0 at extremes
            score = math.log1p(m.volume_24h) * uncertainty
            row = m.to_dict()
            row["uncertainty"] = round(uncertainty, 4)
            row["rank_score"] = round(score, 4)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.sort_values("rank_score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def scan(self) -> pd.DataFrame:
        """
        Full scan: fetch → filter → rank.
        Returns a DataFrame of opportunities sorted by rank_score.
        """
        logger.info("Starting market scan…")
        raw = self._fetch_all()
        filtered = self._filter(raw)

        if not filtered:
            logger.warning("No markets passed filters — returning empty DataFrame.")
            return pd.DataFrame()

        df = self._rank(filtered)
        logger.info(f"Scan complete — {len(df)} markets ready for research.")
        return df
