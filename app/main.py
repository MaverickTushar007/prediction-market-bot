"""
Main Application — FastAPI interface + pipeline orchestration.

Endpoints:
  GET  /health
  POST /pipeline/run          — Run the full pipeline once
  GET  /trades                — List all trades
  GET  /trades/open           — Open positions
  GET  /performance           — Performance metrics
  GET  /risk                  — Risk manager summary
  POST /trades/{market_id}/resolve  — Resolve a trade
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.trade_logger import TradeLogger
from execution.paper_trader import PaperTrader
from execution.trade_executor import TradeExecutor
from prediction.ensemble_model import EnsembleModel
from prediction.probability_model import MarketFeatures
from research.news_scraper import NewsScraper
from research.sentiment_engine import SentimentEngine
from risk.risk_checks import RiskManager
from scanner.market_scanner import MarketScanner
from utils.config import config
from utils.helpers import get_logger, utcnow

logger = get_logger("app.main")

# ── Shared singletons ──────────────────────────────────────────────────────────
scanner = MarketScanner()
scraper = NewsScraper()
sentiment_engine = SentimentEngine(scraper=scraper)
ensemble = EnsembleModel()
risk_manager = RiskManager()
paper_trader = PaperTrader()
trade_logger = TradeLogger()
executor = TradeExecutor(
    risk_manager=risk_manager,
    paper_trader=paper_trader,
    trade_logger=trade_logger,
)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Prediction Market Bot starting up…")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

from signals.api import router as signals_router

app = FastAPI(
    title="Prediction Market Trading Bot",
    description="AI-powered prediction market scanner and paper trading engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(signals_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    max_markets: int = 10
    simulate_resolutions: bool = True


class ResolveRequest(BaseModel):
    outcome: bool


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "paper_trading": config.paper_trading,
        "bankroll": risk_manager.current_bankroll,
        "open_positions": risk_manager.open_position_count,
        "timestamp": utcnow().isoformat(),
    }


# ── Pipeline ───────────────────────────────────────────────────────────────────

@app.post("/pipeline/run", tags=["pipeline"])
def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> Dict[str, Any]:
    """
    Execute the full trading pipeline:
    Scan → Research → Predict → Risk → Execute
    """
    started_at = utcnow().isoformat()
    trades_placed: List[dict] = []
    signals_found = 0
    errors: List[str] = []

    logger.info(f"=== Pipeline run starting (max_markets={cfg.max_markets}) ===")

    # Step 1 — Scan
    try:
        markets_df = scanner.scan()
    except Exception as e:
        raise HTTPException(500, f"Market scan failed: {e}")

    top_markets = markets_df.head(cfg.max_markets)
    logger.info(f"Processing {len(top_markets)} markets…")

    for _, row in top_markets.iterrows():
        market_id = row["market_id"]
        question = row["question"]

        try:
            # Step 2 — Research + Sentiment
            articles = scraper.fetch(question)
            sentiment = sentiment_engine.analyse(question, articles)

            # Step 3 — Probability prediction
            features = MarketFeatures.from_row(
                row.to_dict(),
                sentiment_score=sentiment.score,
                sentiment_confidence=sentiment.confidence,
            )
            news_summary = sentiment.summary
            signal = ensemble.predict(
                features, question=question, news_summary=news_summary,
                market_id=market_id,
            )

            # Log research
            trade_logger.log_research(
                market_id=market_id,
                question=question,
                sentiment_label=sentiment.label,
                sentiment_score=sentiment.score,
                model_prob=signal.model_probability,
                market_price=signal.market_price,
                edge=signal.edge,
            )

            if signal.has_signal:
                signals_found += 1

                # Steps 4+5 — Risk check + Execute
                trade = executor.execute(signal, question=question)

                if trade:
                    trades_placed.append(trade.to_dict())

                    # Optional: simulate resolution for demo
                    if cfg.simulate_resolutions:
                        resolved_trade = executor.resolve(
                            market_id, outcome=(signal.model_probability > 0.5)
                        )
                        if resolved_trade:
                            trades_placed[-1]["pnl"] = resolved_trade.pnl
                            trades_placed[-1]["status"] = "closed"

        except Exception as e:
            err_msg = f"Error processing {market_id}: {traceback.format_exc()}"
            logger.error(err_msg)
            errors.append(str(e))
            continue

    finished_at = utcnow().isoformat()
    trade_logger.log_pipeline_run(
        started_at=started_at,
        finished_at=finished_at,
        markets_scanned=len(top_markets),
        signals_found=signals_found,
        trades_placed=len(trades_placed),
    )

    logger.info(
        f"=== Pipeline complete: scanned={len(top_markets)} "
        f"signals={signals_found} trades={len(trades_placed)} ==="
    )

    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "markets_scanned": len(top_markets),
        "signals_found": signals_found,
        "trades_placed": len(trades_placed),
        "trades": trades_placed,
        "risk": risk_manager.summary(),
        "errors": errors,
    }


# ── Trades ─────────────────────────────────────────────────────────────────────

@app.get("/trades", tags=["trades"])
def list_trades(status: Optional[str] = Query(None, enum=["open", "closed"])):
    """Return all trades from the database."""
    df = trade_logger.get_trades(status=status)
    return {"count": len(df), "trades": df.to_dict(orient="records")}


@app.get("/trades/open", tags=["trades"])
def open_trades():
    """Return currently open paper positions."""
    trades = paper_trader.open_trades
    return {"count": len(trades), "trades": [t.to_dict() for t in trades]}


@app.post("/trades/{market_id}/resolve", tags=["trades"])
def resolve_trade(market_id: str, body: ResolveRequest):
    """Manually resolve an open trade."""
    trade = executor.resolve(market_id, outcome=body.outcome)
    if trade is None:
        raise HTTPException(404, f"No open trade for market {market_id}")
    return {"trade": trade.to_dict(), "risk": risk_manager.summary()}


# ── Performance ────────────────────────────────────────────────────────────────

@app.get("/performance", tags=["analytics"])
def performance():
    """Return performance metrics from closed trades."""
    stats = paper_trader.performance_stats()
    df = trade_logger.get_trades(status="closed")
    stats["trades_in_db"] = len(df)
    return stats


@app.get("/risk", tags=["analytics"])
def risk_summary():
    """Return current risk manager state."""
    return risk_manager.summary()


@app.get("/pipeline/history", tags=["pipeline"])
def pipeline_history():
    """Return log of past pipeline runs."""
    df = trade_logger.get_pipeline_runs()
    return {"runs": df.to_dict(orient="records")}


@app.get("/research", tags=["analytics"])
def research_log():
    """Return market research log."""
    df = trade_logger.get_research()
    return {"count": len(df), "research": df.to_dict(orient="records")}
