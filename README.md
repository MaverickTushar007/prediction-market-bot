# 🤖 Prediction Market Trading Bot

An AI-powered prediction market trading system that scans markets, researches events, estimates probabilities with ML + LLM reasoning, manages risk with Kelly Criterion, and simulates paper trades — all in a clean, modular Python pipeline.

---

## Features

| Module | What it does |
|---|---|
| **Market Scanner** | Fetches Polymarket & Kalshi markets, filters by volume / spread / expiry |
| **Research Engine** | Scrapes RSS feeds and NewsAPI, classifies sentiment (Bullish / Bearish / Neutral) |
| **Probability Model** | Ensemble of Logistic Regression + XGBoost + LLM reasoner |
| **Risk Manager** | Kelly Criterion sizing, max drawdown / daily loss circuit breakers |
| **Paper Trader** | Simulated limit orders, slippage, position tracking, SQLite logging |
| **Plotly Dashboard** | Live equity curve, win rate, Brier score, trade table |
| **FastAPI** | REST API to trigger pipeline runs, query trades, and resolve positions |

---

## Project Structure

```
prediction-market-bot/
├── app/
│   └── main.py               # FastAPI app + pipeline endpoint
├── scanner/
│   └── market_scanner.py     # Polymarket / Kalshi / mock market fetch + filter
├── research/
│   ├── news_scraper.py       # RSS + NewsAPI article fetcher
│   └── sentiment_engine.py   # VADER sentiment classifier
├── prediction/
│   ├── probability_model.py  # LogisticModel, XGBoostModel, LLMReasoner
│   └── ensemble_model.py     # Weighted ensemble + signal generation
├── risk/
│   ├── kelly.py              # Kelly Criterion position sizing
│   └── risk_checks.py        # Drawdown / daily loss / concurrent trade limits
├── execution/
│   ├── paper_trader.py       # Simulated order execution + position tracking
│   └── trade_executor.py     # Orchestrates risk → sizing → execution → logging
├── data/
│   ├── database.py           # SQLite schema + connection
│   └── trade_logger.py       # Persistence layer for trades + research
├── utils/
│   ├── config.py             # Centralised configuration (env vars)
│   └── helpers.py            # Logging, metrics, decorators
├── dashboard/
│   └── dashboard.py          # Plotly Dash performance dashboard
├── tests/
│   └── test_pipeline.py      # pytest test suite (all modules)
├── run_demo.py               # End-to-end demo (no API keys needed)
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourname/prediction-market-bot.git
cd prediction-market-bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the demo (no API keys needed)

```bash
python run_demo.py
```

This uses mock market data and simulated news. You'll see the full pipeline output in the terminal and trades saved to `data/demo_trades.db`.

### 3. Launch the dashboard

```bash
python dashboard/dashboard.py
# Open http://localhost:8050
```

### 4. Launch the FastAPI server

```bash
uvicorn app.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

### 5. Run the test suite

```bash
pytest tests/ -v
```

---

## Configuration

All settings can be overridden via environment variables:

```bash
# API Keys (optional — bot works with mocks if absent)
export POLYMARKET_API_KEY="your_key"
export KALSHI_API_KEY="your_key"
export NEWS_API_KEY="your_key"
export OPENAI_API_KEY="your_key"       # Enables LLM reasoning step

# Risk parameters
export BANKROLL="10000"
export MAX_POSITION_PCT="0.05"
export MAX_CONCURRENT_TRADES="15"
export DAILY_LOSS_LIMIT_PCT="0.15"
export MAX_DRAWDOWN_PCT="0.08"
export KELLY_FRACTION="0.25"

# Signal threshold
export MIN_EDGE="0.04"
```

Or copy `.env.example` to `.env` and use `python-dotenv`.

---

## Pipeline Design

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE (per run)                    │
│                                                         │
│  1. MarketScanner                                       │
│     └─ Fetch Polymarket + Kalshi (mock if no key)       │
│     └─ Filter: volume > 200, days < 30, spread < 5%     │
│     └─ Rank by liquidity × uncertainty score            │
│                                                         │
│  2. NewsScraper + SentimentEngine                       │
│     └─ RSS feeds + NewsAPI                              │
│     └─ VADER sentiment → Bullish / Bearish / Neutral    │
│                                                         │
│  3. EnsembleModel                                       │
│     └─ LogisticRegression (25%)                         │
│     └─ XGBoost (35%)                                    │
│     └─ LLMReasoner / GPT-4o-mini (40%, if key present)  │
│     └─ Signal if edge > 0.04                            │
│                                                         │
│  4. RiskManager                                         │
│     └─ Kelly Criterion sizing (0.25× fractional)        │
│     └─ Max position 5% bankroll                         │
│     └─ Drawdown / daily loss circuit breakers           │
│                                                         │
│  5. PaperTrader + TradeLogger                           │
│     └─ Simulate limit order fill + slippage             │
│     └─ Persist to SQLite                                │
└─────────────────────────────────────────────────────────┘
```

---

## Kelly Criterion

```
f* = (p × b − q) / b

p  = model probability of winning
q  = 1 − p
b  = net odds = (1 − price) / price

Bet size = f* × 0.25 × bankroll   (fractional Kelly)
         capped at 5% of bankroll
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System status + bankroll |
| `POST` | `/pipeline/run` | Run full pipeline |
| `GET` | `/trades` | All trades |
| `GET` | `/trades/open` | Open positions |
| `POST` | `/trades/{id}/resolve` | Resolve a position |
| `GET` | `/performance` | Win rate, Sharpe, Brier |
| `GET` | `/risk` | Risk manager state |
| `GET` | `/research` | Market research log |

---

## Metrics Tracked

- **Win Rate** — % of trades that closed profitably
- **Sharpe Ratio** — risk-adjusted return (annualised)
- **Profit Factor** — gross profit / gross loss
- **Brier Score** — probability calibration (lower = better)
- **Max Drawdown** — peak-to-trough equity decline

---

## Extending the Bot

- **Add a new market source**: implement a client in `scanner/market_scanner.py` returning `List[Market]`
- **Add a new model**: subclass `BaseModel` in `prediction/probability_model.py`
- **Adjust risk rules**: edit thresholds in `utils/config.py` or `risk/risk_checks.py`
- **Live trading**: replace `PaperTrader` with a real API client in `execution/`

---

## Disclaimer

This is a research and educational project. It operates in **paper trading mode only** by default. Do not use this for real financial decisions.
