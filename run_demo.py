"""
run_demo.py — Runs the full prediction market pipeline end-to-end.
No API keys required. Uses mock market data and simulated news.

Usage:
    python run_demo.py
    python run_demo.py --markets 20 --no-resolve
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from scanner.market_scanner import MarketScanner
from research.news_scraper import NewsScraper
from research.sentiment_engine import SentimentEngine
from prediction.ensemble_model import EnsembleModel
from prediction.probability_model import MarketFeatures
from risk.kelly import calculate_bet_size
from risk.risk_checks import RiskManager
from execution.paper_trader import PaperTrader
from execution.trade_executor import TradeExecutor
from data.trade_logger import TradeLogger
from utils.config import config
from utils.helpers import get_logger, utcnow

logger = get_logger("demo")

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner(text: str, colour: str = CYAN) -> None:
    width = 72
    print(f"\n{colour}{BOLD}{'═'*width}{RESET}")
    print(f"{colour}{BOLD}  {text}{RESET}")
    print(f"{colour}{BOLD}{'═'*width}{RESET}\n")


def run_pipeline(max_markets: int = 15, simulate_resolutions: bool = True) -> dict:
    banner("🤖  PREDICTION MARKET TRADING BOT  —  DEMO RUN")

    # ── Init components ────────────────────────────────────────────────────────
    scanner       = MarketScanner(min_volume=50, max_days=30, max_spread=0.08)
    scraper       = NewsScraper(use_rss=True)  # uses NEWS_API_KEY from env
    sentiment_eng = SentimentEngine(scraper=scraper)
    ensemble      = EnsembleModel()
    risk_manager  = RiskManager(bankroll=config.bankroll)
    paper_trader  = PaperTrader()
    trade_logger  = TradeLogger(db_path="data/demo_trades.db")
    executor      = TradeExecutor(
        risk_manager=risk_manager,
        paper_trader=paper_trader,
        trade_logger=trade_logger,
    )

    started_at = utcnow().isoformat()
    signals, trades_placed, researched = [], [], []

    # ══ STEP 1: MARKET SCAN ════════════════════════════════════════════════════
    banner("STEP 1  —  Market Scanner", CYAN)
    markets_df = scanner.scan()
    top = markets_df.head(max_markets)
    print(f"  {GREEN}✓{RESET} {len(markets_df)} markets scanned, top {len(top)} selected\n")
    print(top[["market_id", "question", "yes_price", "volume_24h",
               "days_to_expiry", "spread", "rank_score"]].to_string(index=False))

    # ══ STEP 2-3: RESEARCH + PREDICTION ═══════════════════════════════════════
    banner("STEPS 2-3  —  Research  ×  Prediction", CYAN)

    for _, row in top.iterrows():
        mid   = row["market_id"]
        q     = row["question"]

        # Sentiment
        articles  = scraper.fetch(q)
        sentiment = sentiment_eng.analyse(q, articles)

        # Features + ensemble
        features = MarketFeatures.from_row(
            row.to_dict(),
            sentiment_score=sentiment.score,
            sentiment_confidence=sentiment.confidence,
        )
        signal = ensemble.predict(
            features, question=q,
            news_summary=sentiment.summary,
            market_id=mid,
        )
        researched.append(signal)

        # Log research
        trade_logger.log_research(
            market_id=mid, question=q,
            sentiment_label=sentiment.label,
            sentiment_score=sentiment.score,
            model_prob=signal.model_probability,
            market_price=signal.market_price,
            edge=signal.edge,
        )

        icon = GREEN + "▲" if signal.direction == "YES" else (
               RED   + "▼" if signal.direction == "NO"  else
               YELLOW + "—")
        print(
            f"  {icon}{RESET}  {mid}  |  "
            f"mkt={signal.market_price:.3f}  "
            f"model={signal.model_probability:.3f}  "
            f"edge={signal.edge:+.4f}  "
            f"sent={sentiment.label:<8}  "
            f"→ {BOLD}{signal.direction}{RESET}"
        )

        if signal.has_signal:
            signals.append(signal)

    print(f"\n  {GREEN}✓{RESET} {len(signals)} trade signals generated out of {len(top)} markets")

    # ══ STEP 4-5: RISK + EXECUTION ════════════════════════════════════════════
    banner("STEPS 4-5  —  Risk Management  ×  Execution", CYAN)

    for signal in signals:
        q = next((r["question"] for _, r in top.iterrows()
                  if r["market_id"] == signal.market_id), signal.question)

        # Kelly sizing
        sizing = calculate_bet_size(
            market_id=signal.market_id,
            direction=signal.direction,
            market_price=signal.market_price,
            model_probability=signal.model_probability,
            bankroll=risk_manager.current_bankroll,
        )

        # Risk check
        risk_result = risk_manager.check_trade(sizing.bet_size_usd, signal.market_id)
        if not risk_result:
            print(f"  {RED}✗{RESET}  [{signal.market_id}] BLOCKED — {risk_result.reason}")
            continue

        # Execute
        trade = executor.execute(signal, question=q)
        if trade is None:
            print(f"  {YELLOW}~{RESET}  [{signal.market_id}] Not filled (simulated)")
            continue

        trades_placed.append(trade)
        print(
            f"  {GREEN}✓{RESET}  OPEN  [{trade.trade_id}]  "
            f"{signal.market_id}  {trade.direction}  "
            f"${trade.size_usd:.2f} @ {trade.entry_price:.4f}  "
            f"kelly={sizing.fractional_kelly:.4f}"
        )

        # Simulate resolution
        if simulate_resolutions:
            outcome = signal.model_probability > 0.5
            resolved = executor.resolve(signal.market_id, outcome=outcome)
            if resolved:
                colour = GREEN if resolved.pnl >= 0 else RED
                sign   = "+" if resolved.pnl >= 0 else ""
                print(
                    f"         {colour}RESOLVED{RESET}  "
                    f"outcome={'YES' if outcome else 'NO'}  "
                    f"PnL={colour}{sign}${resolved.pnl:.2f}{RESET}"
                )

    # ══ PERFORMANCE SUMMARY ════════════════════════════════════════════════════
    banner("PERFORMANCE SUMMARY", GREEN)

    stats = paper_trader.performance_stats()
    risk  = risk_manager.summary()

    rows = [
        ("Total Trades",    stats.get("total_trades", 0)),
        ("Win Rate",        f"{stats.get('win_rate', 0):.1%}"),
        ("Total PnL",       f"${stats.get('total_pnl', 0):+.2f}"),
        ("Profit Factor",   f"{stats.get('profit_factor', 0):.3f}"),
        ("Sharpe Ratio",    f"{stats.get('sharpe_ratio', 0):.3f}"),
        ("Brier Score",     stats.get("brier_score") or "N/A"),
        ("Bankroll",        f"${risk['bankroll']:.2f}"),
        ("Drawdown",        f"{risk['current_drawdown_pct']:.2f}%"),
        ("Open Positions",  risk["open_positions"]),
    ]
    for label, val in rows:
        print(f"  {BOLD}{label:<22}{RESET} {val}")

    # Log pipeline run
    finished_at = utcnow().isoformat()
    trade_logger.log_pipeline_run(
        started_at=started_at,
        finished_at=finished_at,
        markets_scanned=len(top),
        signals_found=len(signals),
        trades_placed=len(trades_placed),
        notes="Demo run",
    )

    banner("✅  PIPELINE COMPLETE", GREEN)
    print(f"  Trades saved to:  {GREEN}data/demo_trades.db{RESET}")
    print(f"  Launch dashboard: {GREEN}python dashboard/dashboard.py{RESET}")
    print(f"  Launch API:       {GREEN}uvicorn app.main:app --reload{RESET}\n")

    return {
        "markets_scanned": len(top),
        "signals": len(signals),
        "trades": len(trades_placed),
        "stats": stats,
        "risk": risk,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Market Bot — Demo")
    parser.add_argument("--markets", type=int, default=15,
                        help="Max markets to process (default: 15)")
    parser.add_argument("--no-resolve", action="store_true",
                        help="Don't simulate trade resolution")
    args = parser.parse_args()

    run_pipeline(
        max_markets=args.markets,
        simulate_resolutions=not args.no_resolve,
    )
