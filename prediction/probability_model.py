"""
Probability Model — individual classifiers used in the ensemble.

Three models:
1. LogisticRegression — baseline statistical model
2. XGBoostClassifier — gradient boosted trees
3. LLMReasoner — wraps an LLM (OpenAI / Anthropic) for qualitative reasoning

Each model exposes a `.predict(features) -> float` interface returning
P(YES) ∈ [0, 1].
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from utils.config import config
from utils.helpers import clamp, get_logger

logger = get_logger(__name__)


# ── Feature vector ─────────────────────────────────────────────────────────────

@dataclass
class MarketFeatures:
    """Structured feature vector fed to all ML models."""
    market_price: float          # Current market YES price
    volume_24h: float            # Log-normalised later
    days_to_expiry: float
    spread: float
    sentiment_score: float       # From SentimentEngine [-1,+1]
    sentiment_confidence: float
    uncertainty: float           # 1 - |price - 0.5| * 2
    open_interest: float
    # Category one-hot (encoded externally before passing to array)
    category_economics: float = 0.0
    category_politics: float = 0.0
    category_sports: float = 0.0
    category_crypto: float = 0.0
    category_technology: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for sklearn/xgboost."""
        import math
        return np.array([
            self.market_price,
            math.log1p(self.volume_24h),
            self.days_to_expiry / 30.0,   # normalise to ~[0,1]
            self.spread,
            self.sentiment_score,
            self.sentiment_confidence,
            self.uncertainty,
            math.log1p(self.open_interest),
            self.category_economics,
            self.category_politics,
            self.category_sports,
            self.category_crypto,
            self.category_technology,
        ], dtype=np.float32)

    @classmethod
    def from_row(cls, row: dict, sentiment_score: float = 0.0,
                 sentiment_confidence: float = 0.0) -> "MarketFeatures":
        """Build from a scanner DataFrame row dict."""
        cat = str(row.get("category", "")).lower()
        return cls(
            market_price=float(row.get("mid_price", row.get("yes_price", 0.5))),
            volume_24h=float(row.get("volume_24h", 0)),
            days_to_expiry=float(row.get("days_to_expiry", 7)),
            spread=float(row.get("spread", 0.02)),
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            uncertainty=float(row.get("uncertainty", 0.5)),
            open_interest=float(row.get("open_interest", 0)),
            category_economics=float("economics" in cat),
            category_politics=float("politics" in cat),
            category_sports=float("sports" in cat),
            category_crypto=float("crypto" in cat),
            category_technology=float("technology" in cat),
        )


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseModel(ABC):
    """Abstract base for all probability estimators."""

    @abstractmethod
    def predict(self, features: MarketFeatures) -> float:
        """Return P(YES) ∈ [0, 1]."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── 1. Logistic Regression ────────────────────────────────────────────────────

class LogisticModel(BaseModel):
    """
    Simple logistic regression.
    Pre-trained on synthetic data; re-trains automatically on accumulated
    trade history when enough samples exist in the database.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        self._clf = LogisticRegression(max_iter=1000, C=1.0)
        self._trained = False
        self._init_with_synthetic_data()

    def _init_with_synthetic_data(self, n: int = 2000) -> None:
        """Bootstrap model on synthetic market outcomes."""
        rng = np.random.default_rng(42)
        prices = rng.uniform(0.05, 0.95, n)
        sentiments = rng.uniform(-1, 1, n)
        # Ground truth: market price is a strong prior; sentiment adds noise
        true_p = np.clip(prices + sentiments * 0.08 + rng.normal(0, 0.05, n), 0.0, 1.0)
        outcomes = (rng.uniform(0, 1, n) < true_p).astype(int)

        features = np.column_stack([
            prices,
            rng.lognormal(7, 1.5, n),          # volume
            rng.uniform(1, 30, n) / 30,          # days_to_expiry norm
            rng.uniform(0.002, 0.05, n),         # spread
            sentiments,
            rng.uniform(0.3, 1.0, n),            # confidence
            1 - np.abs(prices - 0.5) * 2,        # uncertainty
            rng.lognormal(6, 1.5, n),            # OI
            *[rng.integers(0, 2, n) for _ in range(5)],  # category OHE
        ]).T
        # Reshape correctly: (n_samples, n_features)
        features = np.column_stack([
            prices,
            np.log1p(rng.lognormal(7, 1.5, n)),
            rng.uniform(1, 30, n) / 30,
            rng.uniform(0.002, 0.05, n),
            sentiments,
            rng.uniform(0.3, 1.0, n),
            1 - np.abs(prices - 0.5) * 2,
            np.log1p(rng.lognormal(6, 1.5, n)),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
        ])

        X_scaled = self._scaler.fit_transform(features)
        self._clf.fit(X_scaled, outcomes)
        self._trained = True
        logger.info("LogisticModel bootstrapped on synthetic data.")

    def predict(self, features: MarketFeatures) -> float:
        X = features.to_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        prob = self._clf.predict_proba(X_scaled)[0][1]
        return clamp(float(prob))


# ── 2. XGBoost model ──────────────────────────────────────────────────────────

class XGBoostModel(BaseModel):
    """
    XGBoost gradient-boosted classifier for probability estimation.
    Falls back to a heuristic if xgboost is not installed.
    """

    def __init__(self):
        self._model = None
        self._trained = False
        self._init_with_synthetic_data()

    def _init_with_synthetic_data(self, n: int = 3000) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed — XGBoostModel will use heuristic fallback.")
            return

        rng = np.random.default_rng(123)
        prices = rng.uniform(0.05, 0.95, n)
        sentiments = rng.uniform(-1, 1, n)
        true_p = np.clip(prices + sentiments * 0.10 + rng.normal(0, 0.06, n), 0.0, 1.0)
        outcomes = (rng.uniform(0, 1, n) < true_p).astype(int)

        X = np.column_stack([
            prices,
            np.log1p(rng.lognormal(7, 1.5, n)),
            rng.uniform(1, 30, n) / 30,
            rng.uniform(0.002, 0.05, n),
            sentiments,
            rng.uniform(0.3, 1.0, n),
            1 - np.abs(prices - 0.5) * 2,
            np.log1p(rng.lognormal(6, 1.5, n)),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
        ])

        self._model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self._model.fit(X, outcomes, verbose=False)
        self._trained = True
        logger.info("XGBoostModel bootstrapped on synthetic data.")

    def predict(self, features: MarketFeatures) -> float:
        if self._model is None:
            # Heuristic fallback: blend market price and sentiment
            return clamp(
                features.market_price + features.sentiment_score * 0.05
            )
        X = features.to_array().reshape(1, -1)
        prob = self._model.predict_proba(X)[0][1]
        return clamp(float(prob))


# ── 3. LLM Reasoner ───────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """You are a calibrated probability forecaster for prediction markets.
Given a market question, relevant news summary, and current market price, estimate the
true probability that the event resolves YES.

Return ONLY a JSON object with exactly these keys:
{
  "probability": <float between 0 and 1>,
  "confidence": <float between 0 and 1>,
  "reasoning": "<one sentence>"
}

Be well-calibrated. If uncertain, stay close to the market price.
Do NOT include markdown or any text outside the JSON."""


class LLMReasoner(BaseModel):
    """
    Uses an LLM (OpenAI GPT or Anthropic Claude) for qualitative reasoning.
    Falls back to market price if API key is missing.
    """

    def __init__(self, api_key: str = config.openai_api_key):
        self.api_key = api_key
        self._last_reasoning: str = ""

    def predict(self, features: MarketFeatures, question: str = "",
                news_summary: str = "") -> float:
        import os
        if not self.api_key and not os.getenv("GROQ_API_KEY",""):
            logger.debug("No LLM API key — returning market price as estimate.")
            return clamp(features.market_price)
        # Always try Groq first regardless of OpenAI key

        prompt = (
            f"Question: {question}\n"
            f"Current market YES price: {features.market_price:.3f}\n"
            f"News summary: {news_summary or 'No recent news.'}\n"
            f"Sentiment: {features.sentiment_score:+.3f} "
            f"({'Bullish' if features.sentiment_score > 0.05 else 'Bearish' if features.sentiment_score < -0.05 else 'Neutral'})\n"
            f"Days to expiry: {features.days_to_expiry:.1f}\n"
        )

        # Try Groq first (free)
        try:
            import os
            from groq import Groq
            groq_key = os.getenv("GROQ_API_KEY", "")
            if groq_key:
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=200, temperature=0.1,
                )
                text = resp.choices[0].message.content or "{}"
                try:
                    data = json.loads(text.strip())
                    prob = clamp(float(data.get("probability", features.market_price)))
                except Exception:
                    prob = clamp(float(text.strip()))
                self._last_reasoning = data.get("reasoning", "") if isinstance(data, dict) else ""
                logger.info(f"Groq LLM estimate: {prob:.3f}")
                return prob
        except Exception as e:
            logger.debug(f"Groq failed: {e}")
        # Fallback: OpenAI
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200, temperature=0.1,
            )
            text = response.choices[0].message.content or "{}"
            data = json.loads(text.strip())
            prob = clamp(float(data.get("probability", features.market_price)))
            self._last_reasoning = data.get("reasoning", "")
            logger.info(f"OpenAI LLM estimate: {prob:.3f} — {self._last_reasoning}")
            return prob
        except Exception as e:
            logger.warning(f"LLM call failed: {e} — falling back to market price.")
            return clamp(features.market_price)

    @property
    def last_reasoning(self) -> str:
        return self._last_reasoning
