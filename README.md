cat > ~/Downloads/prediction-market-bot/README.md << 'READMEEOF'
# 📈 AI-Powered Quantitative Trading Signals

> Real-time trading signals for 86 assets across Crypto, Stocks, ETFs, Indices, Commodities & Forex — powered by an ML ensemble, OpenRouter LLM reasoning, and live RSS news.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🧠 Architecture
```
yfinance (live OHLCV)  +  RSS News (7 feeds)
           ↓                      ↓
   Feature Engineering      Sentiment Scoring
   (21 technical indicators)  (keyword-based)
           ↓                      ↓
   XGBoost + LightGBM Ensemble   OpenRouter LLM
   (5-day swing prediction)      (qualitative reasoning)
           ↓                      ↓
        ML Probability  ←→  AI Direction + Reasoning
                    ↓
         ATR-Based TP/SL + Quant Metrics
         (Kelly Criterion, Expected Value, Sharpe)
                    ↓
         FastAPI Backend → Bloomberg-style Dashboard
```

---

## ✨ Features

### Signal Generation
- **86 tickers** across 6 asset classes: Crypto, Stocks, ETFs, Indices, Commodities, Forex
- **ATR-14 based Take Profit & Stop Loss** — volatility-adjusted levels, not fixed percentages
- **5-day swing trade horizon** with BUY/SELL direction and confidence score

### ML Ensemble Model (`signals/ml_model.py`)
- **XGBoost + LightGBM** trained on 2 years of historical OHLCV data per ticker
- **21 features**: RSI-14, MACD, Bollinger Bands, ATR%, volume ratio, momentum, Stochastic %K, SMA distances, 52-week position
- **Sentiment blending**: 10% weight from RSS news sentiment score
- **Time-series aware split**: no lookahead bias, walk-forward validation
- Feature importance explained per signal (top 3 drivers shown in dashboard)

### Quant Metrics
- **Kelly Criterion**: optimal position sizing based on ML probability and R:R ratio
- **Expected Value**: probability-weighted profit/loss per unit
- **Sharpe Estimate**: risk-adjusted return proxy per trade setup
- **Risk/Reward Ratio**: ATR-calibrated per asset class

### AI Reasoning Layer
- **Primary**: Groq (Llama 3.1) — fastest inference
- **Fallback 1**: OpenRouter (Gemma 3, free tier) — reliable backup
- **Fallback 2**: Rule-based technical analysis — always available
- LLM receives: price data, RSI, MACD, SMA position, top RSS headlines → generates plain-English reasoning

### Live News Pipeline
- **7 RSS feeds**: Yahoo Finance, CNBC, MarketWatch, CryptoNews, Investing.com, SeekingAlpha, FT Markets
- **Single fetch, cached per run** — O(1) per ticker instead of O(n) HTTP calls
- **Keyword + alias matching** per asset (e.g. "bitcoin" → "btc", "ethereum" → "eth")
- **Sentiment scoring**: keyword-based BULLISH/BEARISH/NEUTRAL tagging

### Dashboard (`dashboard/signals_dashboard.html`)
- Bloomberg terminal aesthetic — IBM Plex Mono, dark theme
- **ML probability bar** — visual BUY% vs SELL% split with model agreement
- **Feature importance bars** — top 3 predictive features per signal
- **GBM price simulation** — 30-day Monte Carlo path with TP/SL lines
- Filter by asset class, direction, ML confidence
- Auto-refresh every hour

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/MaverickTushar007/prediction-market-bot.git
cd prediction-market-bot
conda create -n trading python=3.10
conda activate trading
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
export GROQ_API_KEY="your_groq_key"        # Free at console.groq.com
export OPENROUTER_API_KEY="your_key"        # Free at openrouter.ai
```

### 3. Generate Signals
```bash
python -c "
from signals.signal_generator import generate_signals
import json
from pathlib import Path
from datetime import datetime
sigs = generate_signals(max_assets=86)
Path('data/signals_cache.json').write_text(
    json.dumps({'signals': sigs, 'generated_at': datetime.now().isoformat()}, indent=2)
)
print(f'Generated {len(sigs)} signals')
"
```

### 4. Start API
```bash
uvicorn app.main:app --port 8000
```

### 5. Open Dashboard
Open `dashboard/signals_dashboard.html` in your browser.

---

## 📁 Project Structure
```
prediction-market-bot/
├── signals/
│   ├── signal_generator.py   # Main pipeline: price → features → ML → AI → signal
│   ├── ml_model.py           # XGBoost + LightGBM ensemble, ATR features
│   └── api.py                # FastAPI router for signal endpoints
├── research/
│   └── news_scraper.py       # RSS feed aggregator with sentiment scoring
├── app/
│   └── main.py               # FastAPI application
├── dashboard/
│   └── signals_dashboard.html # Bloomberg-style frontend
├── prediction/
│   ├── probability_model.py  # Logistic regression baseline
│   └── ensemble_model.py     # Model ensemble logic
├── execution/
│   └── paper_trader.py       # Paper trading simulation
└── data/
    └── database.py           # SQLite trade logging
```

---

## 🔬 ML Model Details

| Parameter | Value |
|-----------|-------|
| Models | XGBoost + LightGBM (soft voting) |
| Training data | 2 years daily OHLCV |
| Prediction horizon | 5 days |
| Label threshold | +2% return = BUY |
| Train/test split | 80/20 time-series |
| Features | 21 technical indicators |
| Sentiment weight | 10% |

**Feature set**: `ret_1d, ret_3d, ret_5d, ret_10d, ret_20d, rsi, macd, macd_signal, macd_hist, bb_pct, bb_width, atr_pct, vol_ratio, vol_trend, dist_sma20, dist_sma50, high_52w, low_52w, roc_5, roc_10, stoch_k`

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/signals/` | Get all cached signals |
| POST | `/signals/refresh` | Trigger background refresh |
| GET | `/health` | API health check |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Data | yfinance, feedparser |
| ML | XGBoost, LightGBM, scikit-learn |
| AI | Groq (Llama 3.1), OpenRouter (Gemma 3) |
| Backend | FastAPI, uvicorn |
| Frontend | Vanilla JS, Chart.js, IBM Plex Mono |
| Storage | SQLite, JSON cache |

---

*Built as a quantitative finance portfolio project — demonstrating ML ensemble modeling, real-time data pipelines, and institutional-grade signal generation.*
