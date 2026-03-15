"""
Probability Predictor
Uses OpenRouter LLM + news to estimate true probability vs market price.
Calculates edge, EV, Kelly position size.
"""
import os
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemma-3-4b-it:free"


def estimate_probability(question: str, yes_price: float, news_context: str = "") -> Optional[dict]:
    """
    Ask LLM to estimate true probability of a prediction market question.
    Returns estimated probability, confidence, reasoning, and edge.
    """
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None

    prompt = f"""You are a prediction market analyst. Estimate the probability of this event occurring.

Question: {question}
Current market price (YES): {yes_price:.1%}
Recent news context: {news_context[:500] if news_context else "No recent news available."}

Analyze carefully and return ONLY valid JSON:
{{
  "estimated_probability": <float 0-1, your best estimate>,
  "confidence": <"HIGH", "MEDIUM", or "LOW">,
  "reasoning": "<2-3 sentences explaining your estimate>",
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "recommendation": "<BUY_YES, BUY_NO, or PASS>"
}}

Only recommend trading if you see a clear edge vs the market price."""

    try:
        resp = requests.post(
            OPENROUTER_API,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400},
            timeout=20,
        )
        text = resp.text.strip()
        if "{" in text:
            # Parse JSON from response
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            content = content.replace("```json", "").replace("```", "").strip()
            if "{" in content:
                content = content[content.find("{"):content.rfind("}")+1]
            import re
            content = re.sub(r',\s*([}\]])', r'\1', content)
            result = json.loads(content)

            est_prob = float(result.get("estimated_probability", yes_price))
            edge = round(est_prob - yes_price, 3)

            return {
                "estimated_probability": round(est_prob, 3),
                "market_price":          round(yes_price, 3),
                "edge":                  edge,
                "confidence":            result.get("confidence", "LOW"),
                "reasoning":             result.get("reasoning", ""),
                "key_factors":           result.get("key_factors", []),
                "recommendation":        result.get("recommendation", "PASS"),
                "ev":                    round(edge * (1 / yes_price if yes_price > 0 else 1), 3),
            }
    except Exception as e:
        logger.warning(f"Prediction failed for '{question[:40]}': {e}")
        return None


def kelly_size(prob: float, yes_price: float, bankroll: float = 1000,
               fraction: float = 0.25) -> dict:
    """
    Calculate Kelly Criterion position size.
    Uses fractional Kelly (default 25%) for safety.
    """
    if yes_price <= 0 or yes_price >= 1:
        return {"kelly_pct": 0, "position_size": 0, "contracts": 0}

    b = (1 - yes_price) / yes_price  # net odds
    q = 1 - prob
    full_kelly = (prob * b - q) / b
    frac_kelly = max(0, full_kelly * fraction)
    position   = round(bankroll * frac_kelly, 2)

    return {
        "kelly_pct":     round(frac_kelly * 100, 2),
        "position_size": position,
        "contracts":     int(position / yes_price) if yes_price > 0 else 0,
        "full_kelly_pct": round(full_kelly * 100, 2),
    }
