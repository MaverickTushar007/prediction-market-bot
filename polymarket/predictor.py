import os, json, re, logging, requests
from typing import Optional

logger = logging.getLogger(__name__)

def _call_groq(prompt: str) -> Optional[str]:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "max_tokens": 350},
            timeout=20,
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Groq failed: {e}")
    return None

def _call_openrouter(prompt: str) -> Optional[str]:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    for model in ["google/gemma-3-4b-it:free", "mistralai/mistral-small-3.1-24b-instruct:free", "qwen/qwen3-4b:free"]:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 350},
                timeout=20,
            )
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return None

def estimate_probability(question: str, yes_price: float, news_context: str = "") -> Optional[dict]:
    prompt = f"""You are a prediction market analyst. Estimate the true probability of this event.

Question: {question}
Current market YES price: {yes_price:.1%}
Context: {news_context[:300] if news_context else "No recent news."}

Return ONLY valid JSON with no markdown fences:
{{"estimated_probability": 0.5, "confidence": "MEDIUM", "reasoning": "2 sentence analysis here", "key_factors": ["factor1", "factor2"], "recommendation": "PASS"}}

Rules:
- recommendation = BUY_YES if your estimate > market price + 5%
- recommendation = BUY_NO if your estimate < market price - 5%  
- recommendation = PASS otherwise"""

    raw = _call_groq(prompt) or _call_openrouter(prompt)
    if not raw:
        return None
    try:
        text = raw.replace("```json", "").replace("```", "").strip()
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        r = json.loads(text)
        est = max(0.01, min(0.99, float(r.get("estimated_probability", yes_price))))
        edge = round(est - yes_price, 3)
        return {
            "estimated_probability": round(est, 3),
            "market_price":          round(yes_price, 3),
            "edge":                  edge,
            "confidence":            r.get("confidence", "LOW"),
            "reasoning":             r.get("reasoning", ""),
            "key_factors":           r.get("key_factors", []),
            "recommendation":        r.get("recommendation", "PASS"),
            "ev":                    round(edge * (1/yes_price if yes_price > 0 else 1), 3),
        }
    except Exception as e:
        logger.warning(f"Parse failed: {e} | raw: {raw[:100]}")
        return None

def kelly_size(prob: float, yes_price: float, bankroll: float = 1000, fraction: float = 0.25) -> dict:
    if yes_price <= 0 or yes_price >= 1:
        return {"kelly_pct": 0, "position_size": 0, "contracts": 0}
    b = (1 - yes_price) / yes_price
    fk = max(0, (prob * b - (1-prob)) / b * fraction)
    pos = round(bankroll * fk, 2)
    return {"kelly_pct": round(fk*100,2), "position_size": pos, "contracts": int(pos/yes_price), "full_kelly_pct": round(fk/fraction*100,2)}
