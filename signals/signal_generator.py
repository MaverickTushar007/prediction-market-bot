"""
Signal Generator — fetches real prices, real news, Groq reasoning.
Extension of the prediction-market-bot project.
"""
import os
import json
import numpy as np

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional

import yfinance as yf

from research.news_scraper import NewsScraper
from utils.config import config

logger = logging.getLogger(__name__)

TICKERS = [
    # Crypto
    dict(symbol="BTC-USD",  display="BTC/USD",  name="Bitcoin",         type="CRYPTO",    icon="₿"),
    dict(symbol="ETH-USD",  display="ETH/USD",  name="Ethereum",        type="CRYPTO",    icon="Ξ"),
    dict(symbol="SOL-USD",  display="SOL/USD",  name="Solana",          type="CRYPTO",    icon="◎"),
    dict(symbol="BNB-USD",  display="BNB/USD",  name="BNB",             type="CRYPTO",    icon="B"),
    dict(symbol="XRP-USD",  display="XRP/USD",  name="XRP",             type="CRYPTO",    icon="X"),
    dict(symbol="DOGE-USD", display="DOGE/USD", name="Dogecoin",        type="CRYPTO",    icon="D"),
    dict(symbol="ADA-USD",  display="ADA/USD",  name="Cardano",         type="CRYPTO",    icon="A"),
    dict(symbol="AVAX-USD", display="AVAX/USD", name="Avalanche",       type="CRYPTO",    icon="A"),
    dict(symbol="MATIC-USD",display="MATIC",    name="Polygon",         type="CRYPTO",    icon="M"),
    dict(symbol="DOT-USD",  display="DOT/USD",  name="Polkadot",        type="CRYPTO",    icon="D"),
    dict(symbol="LINK-USD", display="LINK/USD", name="Chainlink",       type="CRYPTO",    icon="L"),
    dict(symbol="LTC-USD",  display="LTC/USD",  name="Litecoin",        type="CRYPTO",    icon="L"),
    dict(symbol="ATOM-USD", display="ATOM/USD", name="Cosmos",          type="CRYPTO",    icon="A"),
    dict(symbol="NEAR-USD", display="NEAR/USD", name="NEAR Protocol",   type="CRYPTO",    icon="N"),
    dict(symbol="APT-USD",  display="APT/USD",  name="Aptos",           type="CRYPTO",    icon="A"),
    dict(symbol="ARB-USD",  display="ARB/USD",  name="Arbitrum",        type="CRYPTO",    icon="A"),
    dict(symbol="OP-USD",   display="OP/USD",   name="Optimism",        type="CRYPTO",    icon="O"),
    dict(symbol="INJ-USD",  display="INJ/USD",  name="Injective",       type="CRYPTO",    icon="I"),
    dict(symbol="FET-USD",  display="FET/USD",  name="Fetch.ai",        type="CRYPTO",    icon="F"),
    dict(symbol="RENDER-USD",display="RNDR",    name="Render",          type="CRYPTO",    icon="R"),
    # Indices
    dict(symbol="^GSPC",    display="S&P 500",  name="S&P 500",         type="INDEX",     icon="S"),
    dict(symbol="^IXIC",    display="NASDAQ",   name="Nasdaq",          type="INDEX",     icon="N"),
    dict(symbol="^DJI",     display="DOW",      name="Dow Jones",       type="INDEX",     icon="D"),
    dict(symbol="^RUT",     display="RUT",      name="Russell 2000",    type="INDEX",     icon="R"),
    dict(symbol="^VIX",     display="VIX",      name="Volatility",      type="INDEX",     icon="V"),
    dict(symbol="^FTSE",    display="FTSE 100", name="FTSE 100",        type="INDEX",     icon="F"),
    dict(symbol="^N225",    display="NIKKEI",   name="Nikkei 225",      type="INDEX",     icon="N"),
    dict(symbol="^GDAXI",   display="DAX",      name="German DAX",      type="INDEX",     icon="D"),
    dict(symbol="^HSI",     display="HANG SENG",name="Hang Seng",       type="INDEX",     icon="H"),
    # ETFs
    dict(symbol="SPY",      display="SPY",      name="S&P 500 ETF",     type="ETF",       icon="S"),
    dict(symbol="QQQ",      display="QQQ",      name="Nasdaq ETF",      type="ETF",       icon="Q"),
    dict(symbol="IWM",      display="IWM",      name="Russell ETF",     type="ETF",       icon="I"),
    dict(symbol="GLD",      display="GLD",      name="Gold ETF",        type="ETF",       icon="G"),
    dict(symbol="SLV",      display="SLV",      name="Silver ETF",      type="ETF",       icon="S"),
    dict(symbol="TLT",      display="TLT",      name="20Y Treasury",    type="ETF",       icon="T"),
    dict(symbol="XLE",      display="XLE",      name="Energy ETF",      type="ETF",       icon="E"),
    dict(symbol="XLF",      display="XLF",      name="Financials ETF",  type="ETF",       icon="F"),
    dict(symbol="XLK",      display="XLK",      name="Tech ETF",        type="ETF",       icon="T"),
    dict(symbol="XLV",      display="XLV",      name="Healthcare ETF",  type="ETF",       icon="H"),
    dict(symbol="ARKK",     display="ARKK",     name="ARK Innovation",  type="ETF",       icon="A"),
    dict(symbol="IEMG",     display="IEMG",     name="Emerging Markets",type="ETF",       icon="E"),
    dict(symbol="HYG",      display="HYG",      name="High Yield Bond", type="ETF",       icon="H"),
    # Stocks
    dict(symbol="AAPL",     display="AAPL",     name="Apple",           type="STOCK",     icon="A"),
    dict(symbol="MSFT",     display="MSFT",     name="Microsoft",       type="STOCK",     icon="M"),
    dict(symbol="GOOGL",    display="GOOGL",    name="Alphabet",        type="STOCK",     icon="G"),
    dict(symbol="AMZN",     display="AMZN",     name="Amazon",          type="STOCK",     icon="A"),
    dict(symbol="NVDA",     display="NVDA",     name="Nvidia",          type="STOCK",     icon="N"),
    dict(symbol="TSLA",     display="TSLA",     name="Tesla",           type="STOCK",     icon="T"),
    dict(symbol="META",     display="META",     name="Meta",            type="STOCK",     icon="M"),
    dict(symbol="NFLX",     display="NFLX",     name="Netflix",         type="STOCK",     icon="N"),
    dict(symbol="AMD",      display="AMD",      name="AMD",             type="STOCK",     icon="A"),
    dict(symbol="CRM",      display="CRM",      name="Salesforce",      type="STOCK",     icon="C"),
    dict(symbol="ORCL",     display="ORCL",     name="Oracle",          type="STOCK",     icon="O"),
    dict(symbol="INTC",     display="INTC",     name="Intel",           type="STOCK",     icon="I"),
    dict(symbol="QCOM",     display="QCOM",     name="Qualcomm",        type="STOCK",     icon="Q"),
    dict(symbol="PYPL",     display="PYPL",     name="PayPal",          type="STOCK",     icon="P"),
    dict(symbol="SHOP",     display="SHOP",     name="Shopify",         type="STOCK",     icon="S"),
    dict(symbol="UBER",     display="UBER",     name="Uber",            type="STOCK",     icon="U"),
    dict(symbol="COIN",     display="COIN",     name="Coinbase",        type="STOCK",     icon="C"),
    dict(symbol="PLTR",     display="PLTR",     name="Palantir",        type="STOCK",     icon="P"),
    dict(symbol="MSTR",     display="MSTR",     name="MicroStrategy",   type="STOCK",     icon="M"),
    dict(symbol="JPM",      display="JPM",      name="JPMorgan",        type="STOCK",     icon="J"),
    dict(symbol="GS",       display="GS",       name="Goldman Sachs",   type="STOCK",     icon="G"),
    dict(symbol="BAC",      display="BAC",      name="Bank of America", type="STOCK",     icon="B"),
    dict(symbol="V",        display="VISA",     name="Visa",            type="STOCK",     icon="V"),
    dict(symbol="MA",       display="MA",       name="Mastercard",      type="STOCK",     icon="M"),
    dict(symbol="DIS",      display="DIS",      name="Disney",          type="STOCK",     icon="D"),
    dict(symbol="BABA",     display="BABA",     name="Alibaba",         type="STOCK",     icon="A"),
    dict(symbol="TSM",      display="TSM",      name="TSMC",            type="STOCK",     icon="T"),
    dict(symbol="ASML",     display="ASML",     name="ASML",            type="STOCK",     icon="A"),
    dict(symbol="ABNB",     display="ABNB",     name="Airbnb",          type="STOCK",     icon="A"),
    dict(symbol="SNOW",     display="SNOW",     name="Snowflake",       type="STOCK",     icon="S"),
    dict(symbol="NET",      display="NET",      name="Cloudflare",      type="STOCK",     icon="C"),
    # Commodities
    dict(symbol="GC=F",     display="GOLD",     name="Gold Futures",    type="COMMODITY", icon="G"),
    dict(symbol="CL=F",     display="OIL",      name="Crude Oil",       type="COMMODITY", icon="O"),
    dict(symbol="SI=F",     display="SILVER",   name="Silver Futures",  type="COMMODITY", icon="S"),
    dict(symbol="NG=F",     display="NAT GAS",  name="Natural Gas",     type="COMMODITY", icon="N"),
    dict(symbol="HG=F",     display="COPPER",   name="Copper",          type="COMMODITY", icon="C"),
    dict(symbol="ZW=F",     display="WHEAT",    name="Wheat",           type="COMMODITY", icon="W"),
    # Forex
    dict(symbol="EURUSD=X", display="EUR/USD",  name="Euro/Dollar",     type="FOREX",     icon="€"),
    dict(symbol="GBPUSD=X", display="GBP/USD",  name="Pound/Dollar",    type="FOREX",     icon="£"),
    dict(symbol="USDJPY=X", display="USD/JPY",  name="Dollar/Yen",      type="FOREX",     icon="¥"),
    dict(symbol="AUDUSD=X", display="AUD/USD",  name="Aussie/Dollar",   type="FOREX",     icon="A"),
    dict(symbol="USDCAD=X", display="USD/CAD",  name="Dollar/CAD",      type="FOREX",     icon="C"),
    dict(symbol="USDCHF=X", display="USD/CHF",  name="Dollar/Franc",    type="FOREX",     icon="F"),
    dict(symbol="USDINR=X", display="USD/INR",  name="Dollar/Rupee",    type="FOREX",     icon="₹"),
]


@dataclass
class TradeSignal:
    symbol: str
    display: str
    name: str
    type: str
    icon: str
    direction: str
    entry: float
    target: float
    stop_loss: float
    confidence: int
    sentiment: str
    sentiment_score: float
    ai_reasoning: str
    news: list
    confluences: list
    price_change_pct: float
    volume_vs_avg: float
    generated_at: str
    risk_reward: float = 1.0
    ml_probability: float = 0.0
    ml_confidence: str = ""
    ml_agreement: float = 0.0
    ml_note: str = ""
    top_features: list = None

    def __post_init__(self):
        if self.top_features is None:
            self.top_features = []


def _fetch_price_data(symbol: str) -> Optional[dict]:
    """Fetch OHLCV + basic technicals via yfinance."""
    import time
    time.sleep(0.3)  # avoid yfinance rate limit
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="60d", interval="1d")
        if hist is None or hist.empty:
            # fallback: try info dict
            info = tk.fast_info
            price = float(info.last_price) if hasattr(info, "last_price") and info.last_price else None
            if not price:
                return None
            return {"price": price, "change_pct": 0.0, "high": price, "low": price,
                    "volume": 0, "rsi": 50, "vol_ratio": 1.0, "sma20": price, "above_sma20": True}
        latest   = hist.iloc[-1]
        prev     = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
        price    = float(latest["Close"]) if latest["Close"] is not None else None
        if not price:
            return None
        prev_cls = float(prev["Close"])
        
        # Calculate ATR-14
        highs = hist["High"].astype(float)
        lows  = hist["Low"].astype(float)
        closes = hist["Close"].astype(float)
        tr_list = []
        for i in range(1, len(hist)):
            tr = max(
                highs.iloc[i] - lows.iloc[i],
                abs(highs.iloc[i] - closes.iloc[i-1]),
                abs(lows.iloc[i]  - closes.iloc[i-1])
            )
            tr_list.append(tr)
        atr14 = float(sum(tr_list[-14:]) / min(14, len(tr_list))) if tr_list else price * 0.02
        change   = (price - prev_cls) / prev_cls * 100

        # Simple RSI-14
        closes   = hist["Close"].values
        deltas   = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains    = [d if d > 0 else 0 for d in deltas][-14:]
        losses   = [-d if d < 0 else 0 for d in deltas][-14:]
        avg_gain = sum(gains) / 14 if gains else 0
        avg_loss = sum(losses) / 14 if losses else 0.001
        rsi      = 100 - (100 / (1 + avg_gain / avg_loss))

        # 5-day avg volume vs today
        avg_vol  = hist["Volume"].iloc[-6:-1].mean()
        vol_ratio = float(latest["Volume"]) / avg_vol if avg_vol > 0 else 1.0

        # 20-day simple moving average
        sma20    = float(hist["Close"].tail(20).mean())

        return {
            "price":       price,
            "atr14":       round(atr14, 6),
            "change_pct":  change,
            "high":        float(latest["High"]),
            "low":         float(latest["Low"]),
            "volume":      float(latest["Volume"]),
            "rsi":         rsi,
            "vol_ratio":   vol_ratio,
            "sma20":       sma20,
            "above_sma20": price > sma20,
        }
    except Exception as e:
        logger.warning(f"Price fetch failed for {symbol}: {e}")
        return None
# RSS cache — loaded once per run
_rss_cache: List[dict] = []
_rss_fetched: bool = False

def _load_rss_cache():
    global _rss_cache, _rss_fetched
    if _rss_fetched:
        return
    import feedparser
    FEEDS = [
        ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
        ("CNBC",           "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("CryptoNews",     "https://cryptonews.com/news/feed/"),
        ("Investing.com",  "https://www.investing.com/rss/news.rss"),
        ("SeekingAlpha",   "https://seekingalpha.com/feed.xml"),
    ]
    bull_words = {"surge","rally","soar","jump","gain","rise","high","bull","beat","boost","positive","growth","strong"}
    bear_words = {"fall","drop","crash","plunge","decline","down","bear","loss","weak","risk","fear","sell","low","negative"}
    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:40]:
                raw = entry.get("title","")
                title = str(raw() if callable(raw) else raw or "").strip()
                if not title:
                    continue
                hl = title.lower()
                words = set(hl.split())
                bull_s = len(words & bull_words)
                bear_s = len(words & bear_words)
                sent = "BULLISH" if bull_s > bear_s else "BEARISH" if bear_s > bull_s else "NEUTRAL"
                _rss_cache.append({"headline": title[:120], "source": source, "sentiment": sent, "_hl": hl})
        except Exception:
            continue
    _rss_fetched = True
    logger.info(f"RSS cache loaded: {len(_rss_cache)} articles")



def _fetch_news_for(name: str, display: str) -> List[dict]:
    """Fetch real news via RSS feeds — keyword filtered per asset."""
    import feedparser, time
    FEEDS = [
        ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
        ("CNBC",           "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("CryptoNews",     "https://cryptonews.com/news/feed/"),
        ("Investing.com",  "https://www.investing.com/rss/news.rss"),
        ("SeekingAlpha",   "https://seekingalpha.com/feed.xml"),
        ("FT Markets",     "https://www.ft.com/rss/home/uk"),
    ]
    keywords = [name.lower(), display.lower().replace("/","").replace(" ","")]
    results = []
    try:
        for source, url in FEEDS:
            if len(results) >= 4:
                break
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    raw = entry.get("title", "")
                    title = str(raw() if callable(raw) else raw or "").strip()
                    if any(kw in title.lower() for kw in keywords):
                        # Score sentiment from headline keywords
                        hl = title.lower()
                        bull_words = ["surge","rally","soar","jump","gain","rise","high","bull","beat","boost","up","positive","growth","strong"]
                        bear_words = ["fall","drop","crash","plunge","decline","down","bear","loss","weak","risk","fear","sell","low","negative"]
                        bull_score = sum(1 for w in bull_words if w in hl)
                        bear_score = sum(1 for w in bear_words if w in hl)
                        sent = "BULLISH" if bull_score > bear_score else "BEARISH" if bear_score > bull_score else "NEUTRAL"
                        results.append({
                            "headline": title[:120],
                            "source": source,
                            "sentiment": sent
                        })
                        if len(results) >= 4:
                            break
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"RSS fetch failed for {name}: {e}")

    # Fallback to generic market news if no asset-specific news found
    if not results:
        try:
            feed = feedparser.parse("https://finance.yahoo.com/news/rssindex")
            for entry in feed.entries[:3]:
                raw = entry.get("title", "")
                title = str(raw() if callable(raw) else raw or "").strip()
                if title:
                    results.append({"headline": title[:120], "source": "Yahoo Finance", "sentiment": "NEUTRAL"})
        except Exception:
            pass

    return results[:4]


def _rule_based_signal(meta: dict, price_data: dict, news: List[dict]) -> dict:
    """Pure rule-based fallback when all LLMs fail."""
    rsi = price_data.get("rsi", 50)
    above_sma = price_data.get("above_sma20", True)
    change = price_data.get("change_pct", 0)
    price = price_data["price"]

    # Direction from technicals
    bull_signals = sum([rsi < 45, above_sma, change > 0])
    direction = "BUY" if bull_signals >= 2 else "SELL"
    sentiment = "BULLISH" if bull_signals >= 2 else "BEARISH" if bull_signals == 0 else "NEUTRAL"

    # ATR-based levels (1.5×ATR stop, 2.5×ATR target → 1:1.67 RR minimum)
    atr = float(price_data.get("atr14", price * 0.02)) if isinstance(price_data, dict) else price * 0.02
    if not atr or atr < 1e-9:
        m = {"CRYPTO":0.06,"STOCK":0.04,"ETF":0.03,"INDEX":0.03,"COMMODITY":0.05,"FOREX":0.015}.get(meta["type"],0.04)
        atr = price * m / 2.5
    if direction == "BUY":
        target = round(price + 2.5 * atr, 6)
        stop   = round(price - 1.5 * atr, 6)
    else:
        target = round(price - 2.5 * atr, 6)
        stop   = round(price + 1.5 * atr, 6)
    rr       = round(abs(target - price) / abs(price - stop), 1) if price != stop else 1.0

    reasoning = (
        f"RSI at {rsi:.0f} ({'oversold' if rsi < 40 else 'overbought' if rsi > 60 else 'neutral'}), "
        f"price is {'above' if above_sma else 'below'} 20-day SMA, "
        f"24h change {change:+.2f}%. "
        f"Technical setup favours {direction.lower()} with {bull_signals}/3 bullish signals."
    )

    return {
        "direction": direction,
        "entry": price,
        "target": target,
        "stopLoss": stop,
        "stop_loss": stop,
        "confidence": 45 + bull_signals * 8,
        "sentiment": sentiment,
        "sentimentScore": (bull_signals - 1.5) / 1.5,
        "aiReasoning": reasoning,
        "news": [{"headline": (n.get("headline","") if isinstance(n, dict) else getattr(n,"title",str(n))) or "Market update", "source": (n.get("source","") if isinstance(n, dict) else getattr(n,"source","Unknown")) or "Unknown", "sentiment": "NEUTRAL"} for n in news[:3] if n],
        "confluences": [
            {"name": "RSI (14)",      "value": f"{rsi:.0f}", "signal": "BULLISH" if rsi < 45 else "BEARISH" if rsi > 60 else "NEUTRAL"},
            {"name": "vs 20-day SMA", "value": "above" if above_sma else "below", "signal": "BULLISH" if above_sma else "BEARISH"},
            {"name": "24h Change",    "value": f"{change:+.2f}%", "signal": "BULLISH" if change > 0 else "BEARISH"},
            {"name": "Source",        "value": "Rule-based (LLM unavailable)", "signal": "NEUTRAL"},
        ],
    }


def _try_openrouter(prompt: str) -> Optional[str]:
    """Try OpenRouter — free tier with many models."""
    try:
        import requests
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            return None
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/prediction-market-bot",
            },
            json={
                "model": "google/gemma-3-4b-it:free",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
            },
            timeout=20,
        )
        text = resp.text.strip()
        # OpenRouter sometimes returns whitespace before JSON
        if text.startswith("{") or "choices" in text:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        return None
    except Exception as e:
        logger.debug(f"OpenRouter failed: {e}")
        return None


def _try_together(prompt: str) -> Optional[str]:
    """Try Together AI (free $25 credits on signup)."""
    try:
        import requests
        key = os.getenv("TOGETHER_API_KEY", "")
        if not key:
            return None
        resp = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "meta-llama/Llama-3-8b-chat-hf", "messages": [{"role": "user", "content": prompt}], "max_tokens": 300},
            timeout=15,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Together AI failed: {e}")
        return None


def _try_cohere(prompt: str) -> Optional[str]:
    """Try Cohere (free tier available)."""
    try:
        import cohere
        key = os.getenv("COHERE_API_KEY", "")
        if not key:
            return None
        co = cohere.Client(key)
        resp = co.chat(message=prompt, model="command-r", max_tokens=300)
        return resp.text.strip()
    except Exception as e:
        logger.debug(f"Cohere failed: {e}")
        return None


def _try_huggingface(prompt: str) -> Optional[str]:
    """Try HuggingFace Inference API (free)."""
    try:
        import requests
        key = os.getenv("HF_API_KEY", "")
        if not key:
            return None
        resp = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers={"Authorization": f"Bearer {key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
            timeout=20,
        )
        result = resp.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "").replace(prompt, "").strip()
        return None
    except Exception as e:
        logger.debug(f"HuggingFace failed: {e}")
        return None


def _try_openai(prompt: str) -> Optional[str]:
    """Try OpenAI GPT as fallback."""
    try:
        import openai
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            return None
        client = openai.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.debug(f"OpenAI fallback failed: {e}")
        return None


def _try_groq(prompt: str) -> Optional[str]:
    """Try Groq."""
    try:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return None
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.debug(f"Groq failed: {e}")
        return None


def _groq_signal(meta: dict, price_data: dict, news: List[dict]) -> Optional[dict]:
    """Call LLM with Groq → OpenAI → rule-based fallback chain."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not groq_key and not openai_key:
        return _rule_based_signal(meta, price_data, news)

    news_text = "\n".join(f'- {n["headline"]} ({n["source"]})' for n in news) or "No recent news available."

    prompt = f"""Trade signal for {meta['display']} ({meta['type']}).
Price:{price_data['price']:.4f} Change:{price_data['change_pct']:+.2f}% RSI:{price_data['rsi']:.0f} Vol:{price_data['vol_ratio']:.1f}x SMA20:{"above" if price_data['above_sma20'] else "below"}
News:{news_text[:300]}
Return ONLY this JSON (no other text):
{{"direction":"BUY","entry":{price_data['price']:.4f},"target":0.0,"stopLoss":0.0,"confidence":70,"sentiment":"BULLISH","sentimentScore":0.5,"aiReasoning":"your 2 sentence analysis here","news":[],"confluences":[{{"name":"RSI","value":"{price_data['rsi']:.0f}","signal":"NEUTRAL"}},{{"name":"Trend","value":"above SMA20","signal":"BULLISH"}}]}}"""

    # Try each provider in order
    raw = None
    for provider_name, provider_fn in [
        ("Groq",        _try_groq),
        ("OpenRouter",  _try_openrouter),
        ("Together AI", _try_together),
        ("Cohere",      _try_cohere),
        ("HuggingFace", _try_huggingface),
        ("OpenAI",      _try_openai),
    ]:
        raw = provider_fn(prompt)
        if raw:
            logger.info(f"{provider_name} responded for {meta['display']}")
            break

    if not raw:
        logger.info(f"All LLMs failed — using rule-based for {meta['display']}")
        return _rule_based_signal(meta, price_data, news)

    try:
        text = raw.replace("```json", "").replace("```", "").strip()
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        # Fix trailing commas (common Gemma issue)
        import re
        text = re.sub(r',\s*([}\]])', r'', text)
        return json.loads(text)
    except Exception as e:
        logger.warning(f"JSON parse failed for {meta['display']}: {e} — using rule-based")
        return _rule_based_signal(meta, price_data, news)


def _process_ticker(args):
    """Process single ticker in parallel."""
    import time
    meta, existing = args
    try:
        price_data = _fetch_price_data(meta["symbol"])
        if not price_data:
            return existing.get(meta["symbol"])
        news = _fetch_news_for(meta["name"], meta["display"])
        time.sleep(0.3)
        groq_result = _groq_signal(meta, price_data, news)
        if not groq_result:
            return existing.get(meta["symbol"]) or asdict(TradeSignal(
                symbol=meta["symbol"], display=meta["display"], name=meta["name"],
                type=meta["type"], icon=meta["icon"], direction="BUY",
                entry=float(price_data["price"]), target=0.0, stop_loss=0.0,
                confidence=50, sentiment="NEUTRAL", sentiment_score=0.0,
                ai_reasoning="", news=news[:2], confluences=[],
                price_change_pct=float(price_data.get("change_pct",0)),
                volume_vs_avg=float(price_data.get("vol_ratio",1)),
                generated_at=datetime.now(timezone.utc).isoformat(), risk_reward=1.0,
            ))
        # Merge news
        merged_news = []
        for n in (news[:2] + groq_result.get("news",[])[:2]):
            if isinstance(n, dict):
                h = n.get("headline","")
                if callable(h): h = ""
                h = str(h).strip()
                if h and not h.startswith("<built"):
                    merged_news.append({"headline": h[:120], "source": str(n.get("source","Unknown")), "sentiment": str(n.get("sentiment","NEUTRAL"))})
        price = float(price_data["price"])
        atr   = float(price_data.get("atr14", price * 0.02))
        direction = groq_result.get("direction", "BUY")
        m = {"CRYPTO":0.06,"STOCK":0.04,"ETF":0.03,"INDEX":0.03,"COMMODITY":0.05,"FOREX":0.015}.get(meta["type"],0.04)
        if not atr or atr < 1e-9: atr = price * m / 2.5
        raw_tp = float(groq_result.get("target",0) or 0)
        raw_sl = float(groq_result.get("stopLoss",0) or 0)
        atr_tp = round(price + 2.5*atr if direction=="BUY" else price - 2.5*atr, 6)
        atr_sl = round(price - 1.5*atr if direction=="BUY" else price + 1.5*atr, 6)
        target    = raw_tp if raw_tp > 0.001 else atr_tp
        stop_loss = raw_sl if raw_sl > 0.001 else atr_sl
        rr = round(abs(target-price)/abs(price-stop_loss),1) if price != stop_loss else 1.0
        sig = TradeSignal(
            symbol=meta["symbol"], display=meta["display"], name=meta["name"],
            type=meta["type"], icon=meta["icon"], direction=direction,
            entry=price, target=target, stop_loss=stop_loss,
            confidence=int(groq_result.get("confidence",60)),
            sentiment=groq_result.get("sentiment","NEUTRAL"),
            sentiment_score=float(groq_result.get("sentimentScore",0)),
            ai_reasoning=str(groq_result.get("aiReasoning","")),
            news=merged_news, confluences=groq_result.get("confluences",[]),
            price_change_pct=float(price_data.get("change_pct",0)),
            volume_vs_avg=float(price_data.get("vol_ratio",1)),
            generated_at=datetime.now(timezone.utc).isoformat(), risk_reward=rr,
        )
        # ML prediction
        try:
            from signals.ml_model import predict as ml_predict
            ml_sig = ml_predict(meta["symbol"], float(groq_result.get("sentimentScore", 0)))
            if ml_sig:
                ai_dir = sig.direction
                ml_note = f"ML {'confirms' if ml_sig.direction==ai_dir else 'overrides to'} {ml_sig.direction} ({ml_sig.ml_probability:.0%}, {ml_sig.ml_confidence})"
                sig.direction      = ml_sig.direction
                sig.ml_probability = ml_sig.ml_probability
                sig.ml_confidence  = ml_sig.ml_confidence
                sig.ml_agreement   = ml_sig.model_agreement
                sig.ml_note        = ml_note
                sig.top_features   = list(ml_sig.feature_importance.keys())[:3]
                if ml_sig.direction == ai_dir:
                    sig.confidence = min(95, sig.confidence + 10)
        except Exception as ml_e:
            pass
        return asdict(sig)
    except Exception as e:
        logger.warning(f"Signal failed for {meta['display']}: {e}")
        return existing.get(meta["symbol"])


def generate_signals(max_assets: int = 86) -> List[dict]:
    """Main entry — parallel processing, ~8x faster."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json
    from pathlib import Path

    # Load existing cache
    existing = {}
    cache_file = Path("data/signals_cache.json")
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            existing = {s["symbol"]: s for s in cached.get("signals", [])}
        except Exception:
            pass

    # Pre-load RSS once
    _load_rss_cache()

    tickers = TICKERS[:max_assets]
    args = [(meta, existing) for meta in tickers]
    results = {}

    # Phase 1: fetch all prices in parallel (fast, no rate limit)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    price_data_map = {}
    def fetch_price(meta):
        return meta["symbol"], _fetch_price_data(meta["symbol"])
    with ThreadPoolExecutor(max_workers=10) as executor:
        for sym, pd in executor.map(fetch_price, tickers):
            if pd:
                price_data_map[sym] = pd

    # Phase 2: AI calls sequentially (avoid rate limits)
    import time
    for meta in tickers:
        sym = meta["symbol"]
        if sym not in price_data_map:
            if sym in existing:
                results[sym] = existing[sym]
            continue
        try:
            price_data = price_data_map[sym]
            news = _fetch_news_for(meta["name"], meta["display"])
            groq_result = _groq_signal(meta, price_data, news)
            if not groq_result:
                if sym in existing:
                    results[sym] = existing[sym]
                    continue
                groq_result = _rule_based_signal(meta, price_data, news)

            # ML model prediction — blends with AI signal
            try:
                from signals.ml_model import predict as ml_predict
                sentiment_score = float(groq_result.get("sentimentScore", 0))
                ml_sig = ml_predict(sym, sentiment_score)
                if ml_sig:
                    # If ML and AI agree → boost confidence
                    # If they disagree → use ML direction (more data-driven)
                    ai_dir = groq_result.get("direction", "BUY")
                    if ml_sig.direction == ai_dir:
                        groq_result["confidence"] = min(95, int(groq_result.get("confidence", 60)) + 10)
                        groq_result["mlNote"] = f"ML confirms {ai_dir} ({ml_sig.ml_probability:.0%} prob, {ml_sig.ml_confidence} confidence)"
                    else:
                        groq_result["direction"] = ml_sig.direction
                        groq_result["mlNote"] = f"ML overrides to {ml_sig.direction} ({ml_sig.ml_probability:.0%} prob, {ml_sig.ml_confidence} confidence)"
                    groq_result["mlProbability"] = ml_sig.ml_probability
                    groq_result["mlConfidence"] = ml_sig.ml_confidence
                    groq_result["mlAgreement"] = ml_sig.model_agreement
                    groq_result["topFeatures"] = list(ml_sig.feature_importance.keys())[:3]
            except Exception as e:
                logger.debug(f"ML prediction skipped for {meta['display']}: {e}")
            # Build signal
            merged_news = []
            for n in (news[:2] + groq_result.get("news",[])[:2]):
                if isinstance(n, dict):
                    h = n.get("headline","")
                    if callable(h): h = ""
                    h = str(h).strip()
                    if h and not h.startswith("<built"):
                        merged_news.append({"headline": h[:120], "source": str(n.get("source","Unknown")), "sentiment": str(n.get("sentiment","NEUTRAL"))})
            price = float(price_data["price"])
            atr   = float(price_data.get("atr14", price * 0.02))
            direction = groq_result.get("direction","BUY")
            m = {"CRYPTO":0.06,"STOCK":0.04,"ETF":0.03,"INDEX":0.03,"COMMODITY":0.05,"FOREX":0.015}.get(meta["type"],0.04)
            if not atr or atr < 1e-9: atr = price * m / 2.5
            raw_tp = float(groq_result.get("target",0) or 0)
            raw_sl = float(groq_result.get("stopLoss",0) or 0)
            atr_tp = round(price + 2.5*atr if direction=="BUY" else price - 2.5*atr, 6)
            atr_sl = round(price - 1.5*atr if direction=="BUY" else price + 1.5*atr, 6)
            target    = raw_tp if raw_tp > 0.001 else atr_tp
            stop_loss = raw_sl if raw_sl > 0.001 else atr_sl
            rr = round(abs(target-price)/abs(price-stop_loss),1) if price != stop_loss else 1.0
            sig = TradeSignal(
                symbol=sym, display=meta["display"], name=meta["name"],
                type=meta["type"], icon=meta["icon"], direction=direction,
                entry=price, target=target, stop_loss=stop_loss,
                confidence=int(groq_result.get("confidence",60)),
                sentiment=groq_result.get("sentiment","NEUTRAL"),
                sentiment_score=float(groq_result.get("sentimentScore",0)),
                ai_reasoning=str(groq_result.get("aiReasoning","")),
                news=merged_news, confluences=groq_result.get("confluences",[]),
                price_change_pct=float(price_data.get("change_pct",0)),
                volume_vs_avg=float(price_data.get("vol_ratio",1)),
                generated_at=datetime.now(timezone.utc).isoformat(),
                risk_reward=rr,
                ml_probability=float(groq_result.get("mlProbability",0)),
                ml_confidence=str(groq_result.get("mlConfidence","")),
                ml_agreement=float(groq_result.get("mlAgreement",0)),
                ml_note=str(groq_result.get("mlNote","")),
                top_features=groq_result.get("topFeatures",[]),
            )
            # ML prediction
            try:
                from signals.ml_model import predict as ml_predict
                ml_sig = ml_predict(sym, float(groq_result.get("sentimentScore", 0)))
                if ml_sig:
                    ai_dir = sig.direction
                    if ml_sig.direction == ai_dir:
                        sig = sig.__class__(**{**sig.__dict__,
                            "confidence": min(95, sig.confidence + 10),
                            "ml_probability": ml_sig.ml_probability,
                            "ml_confidence": ml_sig.ml_confidence,
                            "ml_agreement": ml_sig.model_agreement,
                            "ml_note": f"ML confirms {ai_dir} ({ml_sig.ml_probability:.0%}, {ml_sig.ml_confidence})",
                            "top_features": list(ml_sig.feature_importance.keys())[:3],
                        })
                    else:
                        sig = sig.__class__(**{**sig.__dict__,
                            "direction": ml_sig.direction,
                            "ml_probability": ml_sig.ml_probability,
                            "ml_confidence": ml_sig.ml_confidence,
                            "ml_agreement": ml_sig.model_agreement,
                            "ml_note": f"ML overrides to {ml_sig.direction} ({ml_sig.ml_probability:.0%}, {ml_sig.ml_confidence})",
                            "top_features": list(ml_sig.feature_importance.keys())[:3],
                        })
            except Exception as e:
                logger.debug(f"ML skipped for {sym}: {e}")
            results[sym] = asdict(sig)
            time.sleep(0.4)
        except Exception as e:
            logger.warning(f"Signal failed for {meta['display']}: {e}")
            if sym in existing:
                results[sym] = existing[sym]

    # Preserve original order
    signals = [results[t["symbol"]] for t in tickers if t["symbol"] in results]
    logger.info(f"Generated {len(signals)} signals (parallel)")
    return signals


