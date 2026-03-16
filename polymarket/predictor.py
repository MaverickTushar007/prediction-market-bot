"""
Multi-Model Voting Predictor
3 models vote independently, weighted consensus drives the decision.
Only trades when models AGREE and edge > 4%.

Model 1: Groq/Llama3.1    → weight 40% (fastest, most reliable)
Model 2: OpenRouter/Gemma  → weight 35% (backup AI)
Model 3: Rule-based logic  → weight 25% (deterministic, never fails)
"""
import os, json, re, logging, requests
from typing import Optional

logger = logging.getLogger(__name__)

# ── MODEL WEIGHTS ──
WEIGHTS = {
    "groq":       0.40,
    "openrouter": 0.35,
    "rules":      0.25,
}

MIN_EDGE       = 0.04   # minimum edge to trade
CONSENSUS_MIN  = 0.60   # at least 60% weighted agreement needed


# ── MODEL 1: GROQ ──
def _call_groq(prompt: str) -> Optional[str]:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300},
            timeout=20,
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Groq failed: {e}")
    return None


# ── MODEL 2: OPENROUTER ──
def _call_openrouter(prompt: str) -> Optional[str]:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    for model in [
        "google/gemma-3-4b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "qwen/qwen3-4b:free",
    ]:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 300},
                timeout=20,
            )
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return None


# ── MODEL 3: RULE-BASED (always works, no API needed) ──
def _rule_based_estimate(question: str, yes_price: float, news_context: str) -> dict:
    """
    Deterministic probability estimate based on:
    - Market price as base
    - News sentiment adjustment
    - Question keyword analysis
    """
    q = question.lower()
    ctx = news_context.lower() if news_context else ""

    # Start from market price
    est = yes_price

    # Positive keywords → nudge up
    bull_words = ["win", "pass", "approve", "rise", "increase", "reach", "hit", "above", "beat"]
    bear_words = ["lose", "fail", "reject", "fall", "decline", "below", "miss", "crash", "drop"]

    bull_q = sum(1 for w in bull_words if w in q)
    bear_q = sum(1 for w in bear_words if w in q)

    # News sentiment
    bull_n = sum(1 for w in bull_words if w in ctx)
    bear_n = sum(1 for w in bear_words if w in ctx)

    # Adjust
    net = (bull_q + bull_n * 0.5) - (bear_q + bear_n * 0.5)
    adjustment = net * 0.02  # max ~10% shift
    est = max(0.05, min(0.95, est + adjustment))

    edge = round(est - yes_price, 3)
    if edge > MIN_EDGE:
        rec = "BUY_YES"
    elif edge < -MIN_EDGE:
        rec = "BUY_NO"
    else:
        rec = "PASS"

    return {
        "estimated_probability": round(est, 3),
        "edge":                  edge,
        "recommendation":        rec,
        "reasoning":             f"Rule-based: market={yes_price:.0%} adj={adjustment:+.3f} net_signal={net:+.1f}",
        "confidence":            "MEDIUM" if abs(net) > 1 else "LOW",
        "model":                 "rules",
    }


# ── JSON PARSER ──
def _parse_json(raw: str, yes_price: float) -> Optional[dict]:
    try:
        text = raw.replace("```json", "").replace("```", "").strip()
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        r = json.loads(text)
        est = max(0.01, min(0.99, float(r.get("estimated_probability", yes_price))))
        edge = round(est - yes_price, 3)
        if edge > MIN_EDGE:
            rec = "BUY_YES"
        elif edge < -MIN_EDGE:
            rec = "BUY_NO"
        else:
            rec = "PASS"
        return {
            "estimated_probability": round(est, 3),
            "edge":                  edge,
            "recommendation":        rec,
            "reasoning":             r.get("reasoning", ""),
            "confidence":            r.get("confidence", "LOW"),
            "key_factors":           r.get("key_factors", []),
        }
    except Exception as e:
        logger.debug(f"JSON parse failed: {e}")
        return None


# ── VOTING ENGINE ──
def _vote(votes: list) -> dict:
    """
    Weighted voting across all models.
    Returns consensus probability and agreement score.
    """
    if not votes:
        return {}

    # Weighted average probability
    total_weight = sum(v["weight"] for v in votes)
    consensus_prob = sum(v["prob"] * v["weight"] for v in votes) / total_weight

    # Count directional agreement
    buy_weight  = sum(v["weight"] for v in votes if v["rec"] == "BUY_YES")
    sell_weight = sum(v["weight"] for v in votes if v["rec"] == "BUY_NO")
    pass_weight = sum(v["weight"] for v in votes if v["rec"] == "PASS")

    # Dominant recommendation
    max_w = max(buy_weight, sell_weight, pass_weight)
    if max_w == buy_weight:
        final_rec = "BUY_YES"
    elif max_w == sell_weight:
        final_rec = "BUY_NO"
    else:
        final_rec = "PASS"

    # Agreement score — how much weight agrees on the dominant side
    agreement = round(max_w / total_weight, 3)

    # Override to PASS if agreement below threshold
    if agreement < CONSENSUS_MIN:
        final_rec = "PASS"
        logger.info(f"Low consensus ({agreement:.0%}) → PASS")

    return {
        "consensus_prob": round(consensus_prob, 3),
        "final_rec":      final_rec,
        "agreement":      agreement,
        "buy_weight":     round(buy_weight, 3),
        "sell_weight":    round(sell_weight, 3),
        "models_voted":   len(votes),
    }


# ── MAIN ENTRY ──
def estimate_probability(question: str, yes_price: float,
                         news_context: str = "") -> Optional[dict]:
    """
    Multi-model voting prediction.
    All 3 models vote, weighted consensus drives the final signal.
    """
    prompt = f"""You are a prediction market analyst. Estimate the true probability of this event.

Question: {question}
Current market YES price: {yes_price:.1%}
Context: {news_context[:300] if news_context else "No recent news."}

Return ONLY valid JSON, no markdown:
{{"estimated_probability": 0.5, "confidence": "MEDIUM", "reasoning": "2 sentence analysis", "key_factors": ["factor1"]}}

Be objective. Base estimate on facts, not the market price."""

    votes = []

    # Vote 1: Groq
    raw_groq = _call_groq(prompt)
    if raw_groq:
        parsed = _parse_json(raw_groq, yes_price)
        if parsed:
            votes.append({
                "model":    "groq",
                "weight":   WEIGHTS["groq"],
                "prob":     parsed["estimated_probability"],
                "rec":      parsed["recommendation"],
                "reasoning": parsed.get("reasoning", ""),
                "confidence": parsed.get("confidence", "LOW"),
            })
            logger.info(f"Groq vote: {parsed['estimated_probability']:.0%} → {parsed['recommendation']}")

    # Vote 2: OpenRouter
    raw_or = _call_openrouter(prompt)
    if raw_or:
        parsed = _parse_json(raw_or, yes_price)
        if parsed:
            votes.append({
                "model":    "openrouter",
                "weight":   WEIGHTS["openrouter"],
                "prob":     parsed["estimated_probability"],
                "rec":      parsed["recommendation"],
                "reasoning": parsed.get("reasoning", ""),
                "confidence": parsed.get("confidence", "LOW"),
            })
            logger.info(f"OpenRouter vote: {parsed['estimated_probability']:.0%} → {parsed['recommendation']}")

    # Vote 3: Rule-based (always runs)
    rules = _rule_based_estimate(question, yes_price, news_context)
    votes.append({
        "model":    "rules",
        "weight":   WEIGHTS["rules"],
        "prob":     rules["estimated_probability"],
        "rec":      rules["recommendation"],
        "reasoning": rules["reasoning"],
        "confidence": rules["confidence"],
    })
    logger.info(f"Rules vote: {rules['estimated_probability']:.0%} → {rules['recommendation']}")

    if not votes:
        return None

    # Get consensus
    consensus = _vote(votes)
    consensus_prob = consensus["consensus_prob"]
    final_rec      = consensus["final_rec"]
    edge           = round(consensus_prob - yes_price, 3)

    # Build reasoning from all models
    reasoning_parts = [f"[{v['model'].upper()}] {v['reasoning'][:80]}" for v in votes if v.get('reasoning')]
    combined_reasoning = " | ".join(reasoning_parts)

    # Key factors from AI models
    key_factors = []
    for v in votes:
        if v.get("model") != "rules":
            key_factors.extend(v.get("key_factors", []))
    key_factors = list(dict.fromkeys(key_factors))[:4]  # dedupe, max 4

    return {
        "estimated_probability": consensus_prob,
        "market_price":          round(yes_price, 3),
        "edge":                  edge,
        "confidence":            "HIGH" if consensus["agreement"] > 0.8 else "MEDIUM" if consensus["agreement"] > 0.6 else "LOW",
        "reasoning":             combined_reasoning[:300],
        "key_factors":           key_factors,
        "recommendation":        final_rec,
        "ev":                    round(edge * (1/yes_price if yes_price > 0 else 1), 3),
        "models_voted":          consensus["models_voted"],
        "agreement":             consensus["agreement"],
        "votes":                 votes,
    }


def kelly_size(prob: float, yes_price: float,
               bankroll: float = 1000, fraction: float = 0.25) -> dict:
    if yes_price <= 0 or yes_price >= 1:
        return {"kelly_pct": 0, "position_size": 0, "contracts": 0}
    b  = (1 - yes_price) / yes_price
    fk = max(0, (prob * b - (1-prob)) / b * fraction)
    pos = round(bankroll * fk, 2)
    return {
        "kelly_pct":      round(fk*100, 2),
        "position_size":  pos,
        "contracts":      int(pos/yes_price),
        "full_kelly_pct": round(fk/fraction*100, 2),
    }
