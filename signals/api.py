"""FastAPI router for trading signals."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/signals", tags=["signals"])

CACHE_FILE = Path("data/signals_cache.json")
_cache = {"signals": [], "generated_at": None}


def _load_cache():
    global _cache
    if CACHE_FILE.exists():
        try:
            _cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            pass


def _save_cache(signals):
    global _cache
    _cache = {"signals": signals, "generated_at": datetime.now(timezone.utc).isoformat()}
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_cache, indent=2))


def _refresh_signals():
    from signals.signal_generator import generate_signals
    sigs = generate_signals()
    _save_cache(sigs)


_load_cache()


@router.get("/")
def get_signals():
    signals = _cache.get("signals", [])
    # Fix any 0.0 target/stop_loss values
    multipliers = {"CRYPTO":0.06,"STOCK":0.04,"ETF":0.03,"INDEX":0.03,"COMMODITY":0.05,"FOREX":0.015}
    for s in signals:
        price = float(s.get("entry", 0) or 0)
        m = multipliers.get(s.get("type","STOCK"), 0.04)
        direction = s.get("direction", "BUY")
        if not s.get("target") or float(s.get("target",0)) < 0.0001:
            s["target"] = round(price*(1+m) if direction=="BUY" else price*(1-m), 4)
        if not s.get("stop_loss") or float(s.get("stop_loss",0)) < 0.0001:
            s["stop_loss"] = round(price*(1-m/2) if direction=="BUY" else price*(1+m/2), 4)
        # Add riskReward
        tp = float(s.get("target",0))
        sl = float(s.get("stop_loss",0))
        if price and sl and price != sl:
            s["riskReward"] = round(abs(tp-price)/abs(price-sl), 1)
    return JSONResponse({"signals": signals, "generated_at": _cache.get("generated_at")})


@router.get("/backtest")
def get_backtest():
    """Run backtest on top 10 tickers and return results."""
    from signals.backtest import run_backtest
    from signals.signal_generator import TICKERS
    import json
    from pathlib import Path

    cache_file = Path("data/backtest_cache.json")
    # Use cache if less than 24 hours old
    if cache_file.exists():
        import time
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            return json.loads(cache_file.read_text())

    # Run backtest on all tickers
    from signals.signal_generator import TICKERS
    symbols = [t["symbol"] for t in TICKERS]
    results = run_backtest(symbols, max_tickers=len(symbols))
    output = {
        "results": results,
        "generated_at": __import__("datetime").datetime.now().isoformat()
    }
    cache_file.write_text(json.dumps(output, indent=2))
    return output


@router.post("/refresh")
def refresh_signals(background_tasks: BackgroundTasks):
    background_tasks.add_task(_refresh_signals)
    return {"status": "refresh started"}
