"""FastAPI router for Polymarket prediction markets."""
import json
import logging
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/polymarket", tags=["polymarket"])
logger = logging.getLogger(__name__)
CACHE_FILE = Path("data/polymarket_cache.json")


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {"markets": [], "generated_at": None}


def _save_cache(data: dict):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2))


@router.get("/markets")
def get_markets():
    """Get scanned + analyzed prediction markets."""
    cache = _load_cache()
    return JSONResponse(cache)


@router.post("/scan")
def scan_and_analyze():
    """Scan Polymarket, analyze top opportunities with AI."""
    from polymarket.scanner import scan_markets
    from polymarket.predictor import estimate_probability, kelly_size
    from research.news_scraper import NewsScraper

    try:
        markets = scan_markets(limit=100)[:20]  # top 20 by volume
        scraper = NewsScraper(use_rss=True, news_api_key="")
        results = []

        for m in markets[:15]:
            # Fetch news for context
            try:
                articles = scraper.fetch(m["question"][:50], max_articles=3)
                news_ctx = " | ".join([
                    str(a.title if hasattr(a, "title") else a)[:100]
                    for a in articles[:3]
                ])
            except Exception:
                news_ctx = ""

            # AI probability estimate
            pred = estimate_probability(m["question"], m["yes_price"], news_ctx)
            if not pred:
                pred = {
                    "estimated_probability": m["yes_price"],
                    "edge": 0,
                    "confidence": "LOW",
                    "reasoning": "No AI estimate available.",
                    "key_factors": [],
                    "recommendation": "PASS",
                    "ev": 0,
                }

            # Kelly sizing (assume $1000 bankroll)
            if pred["recommendation"] in ("BUY_YES", "BUY_NO"):
                price = m["yes_price"] if pred["recommendation"] == "BUY_YES" else m["no_price"]
                sizing = kelly_size(pred["estimated_probability"], price)
            else:
                sizing = {"kelly_pct": 0, "position_size": 0, "contracts": 0}

            results.append({**m, **pred, "sizing": sizing, "news_context": news_ctx})

        from datetime import datetime
        data = {"markets": results, "generated_at": datetime.now().isoformat()}
        _save_cache(data)
        return JSONResponse(data)

    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return JSONResponse({"error": str(e), "markets": []}, status_code=500)
