"""
Ensemble Model — Step 3 of the pipeline.

Combines LogisticModel, XGBoostModel, and LLMReasoner into a weighted
ensemble. Calculates edge vs. market price and generates trade signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from prediction.probability_model import (
    LLMReasoner,
    LogisticModel,
    MarketFeatures,
    XGBoostModel,
)
from utils.config import config
from utils.helpers import clamp, get_logger

logger = get_logger(__name__)


# ── Signal ─────────────────────────────────────────────────────────────────────

@dataclass
class PredictionSignal:
    """Output of the ensemble model for a single market."""
    market_id: str
    question: str
    market_price: float      # Current market YES price
    model_probability: float  # Ensemble P(YES) estimate
    edge: float              # model_probability - market_price
    direction: str           # "YES" | "NO" | "PASS"
    confidence: float        # Model agreement / confidence
    logistic_prob: float
    xgb_prob: float
    llm_prob: float
    llm_reasoning: str = ""

    @property
    def has_signal(self) -> bool:
        return self.direction != "PASS"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── Ensemble ───────────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Weighted ensemble of three probability estimators.

    Default weights: LR=0.25, XGB=0.35, LLM=0.40
    (LLM weighted higher as it understands context,
     but only active when API key is present).

    When LLM is inactive, weights rebalance to LR=0.40, XGB=0.60.
    """

    def __init__(
        self,
        w_logistic: float = 0.25,
        w_xgb: float = 0.35,
        w_llm: float = 0.40,
        min_edge: float = config.min_edge,
    ):
        self.w_logistic = w_logistic
        self.w_xgb = w_xgb
        self.w_llm = w_llm
        self.min_edge = min_edge

        self._logistic = LogisticModel()
        self._xgb = XGBoostModel()
        self._llm = LLMReasoner()

        logger.info("EnsembleModel initialised (LR + XGB + LLM).")

    def predict(
        self,
        features: MarketFeatures,
        question: str = "",
        news_summary: str = "",
        market_id: str = "unknown",
    ) -> PredictionSignal:
        """Generate a calibrated probability estimate and trade signal."""

        # ── Individual predictions ─────────────────────────────────────────
        lr_prob = self._logistic.predict(features)
        xgb_prob = self._xgb.predict(features)
        llm_prob = self._llm.predict(features, question=question, news_summary=news_summary)
        llm_active = bool(config.openai_api_key)

        # ── Weighted average ───────────────────────────────────────────────
        if llm_active:
            total_w = self.w_logistic + self.w_xgb + self.w_llm
            ensemble_prob = (
                self.w_logistic * lr_prob
                + self.w_xgb * xgb_prob
                + self.w_llm * llm_prob
            ) / total_w
        else:
            # Rebalance without LLM
            w_lr, w_xgb = 0.40, 0.60
            ensemble_prob = (w_lr * lr_prob + w_xgb * xgb_prob)

        ensemble_prob = clamp(ensemble_prob)

        # ── Confidence = agreement across models ───────────────────────────
        probs = [lr_prob, xgb_prob, llm_prob]
        std = float(np.std(probs))
        confidence = clamp(1.0 - std * 3)  # High agreement → high confidence

        # ── Edge calculation ───────────────────────────────────────────────
        market_price = features.market_price
        edge_yes = ensemble_prob - market_price
        edge_no = (1 - ensemble_prob) - (1 - market_price)   # = -edge_yes

        best_edge = edge_yes if abs(edge_yes) >= abs(edge_no) else edge_no
        direction: str

        if edge_yes > self.min_edge:
            direction = "YES"
        elif edge_no > self.min_edge:
            direction = "NO"
        else:
            direction = "PASS"

        logger.debug(
            f"[{market_id}] LR={lr_prob:.3f} XGB={xgb_prob:.3f} "
            f"LLM={llm_prob:.3f} → ensemble={ensemble_prob:.3f} "
            f"edge_yes={edge_yes:+.4f} signal={direction}"
        )

        return PredictionSignal(
            market_id=market_id,
            question=question,
            market_price=market_price,
            model_probability=round(ensemble_prob, 4),
            edge=round(best_edge, 4),
            direction=direction,
            confidence=round(confidence, 4),
            logistic_prob=round(lr_prob, 4),
            xgb_prob=round(xgb_prob, 4),
            llm_prob=round(llm_prob, 4),
            llm_reasoning=self._llm.last_reasoning,
        )
