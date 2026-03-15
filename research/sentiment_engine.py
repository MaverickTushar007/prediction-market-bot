"""
Sentiment Engine — Step 2 of the pipeline.

Classifies text sentiment (Bullish/Bearish/Neutral) using a VADER-based
lexicon model (fast, no GPU needed) with an optional LLM upgrade path.
Also aggregates Reddit/social signals when available.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Optional

from research.news_scraper import Article, NewsScraper
from utils.config import config
from utils.helpers import get_logger

logger = get_logger(__name__)


# ── Sentiment result ──────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    label: str           # "Bullish" | "Bearish" | "Neutral"
    score: float         # +1 = very bullish, -1 = very bearish
    confidence: float    # 0-1
    article_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    summary: str

    @property
    def sentiment_probability_boost(self) -> float:
        """
        Convert sentiment score to a probability nudge.
        Positive score adds a small boost to YES probability.
        Range roughly -0.10 to +0.10.
        """
        return round(self.score * 0.10, 4)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── VADER-based classifier ────────────────────────────────────────────────────

# Financial domain word lists — extend as needed
_BULLISH_WORDS = {
    "rise", "surge", "rally", "gain", "beat", "strong", "positive",
    "growth", "increase", "higher", "upbeat", "confident", "optimistic",
    "likely", "expected", "probable", "win", "victory", "succeed",
    "approve", "pass", "confirmed", "breakthrough", "record",
}
_BEARISH_WORDS = {
    "fall", "drop", "decline", "miss", "weak", "negative", "loss",
    "decrease", "lower", "downbeat", "uncertain", "unlikely", "fail",
    "reject", "cancel", "postpone", "disappoint", "concern", "risk",
    "doubt", "skeptical", "oppose", "defeat", "withdraw",
}


def _simple_lexicon_score(text: str) -> float:
    """
    Rule-based sentiment score using financial domain lexicon.
    Returns a score in [-1, +1].
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    bullish = sum(1 for t in tokens if t in _BULLISH_WORDS)
    bearish = sum(1 for t in tokens if t in _BEARISH_WORDS)
    total = bullish + bearish
    if total == 0:
        return 0.0
    return (bullish - bearish) / total


def _vader_score(text: str) -> Optional[float]:
    """Use VADER if available — returns compound score [-1, +1]."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]
    except ImportError:
        return None


def _classify_score(score: float) -> str:
    if score > 0.05:
        return "Bullish"
    if score < -0.05:
        return "Bearish"
    return "Neutral"


def _score_article(article: Article) -> float:
    text = article.text[:1000]
    vader = _vader_score(text)
    if vader is not None:
        return vader
    return _simple_lexicon_score(text)


# ── Reddit fetcher (optional) ─────────────────────────────────────────────────

def _fetch_reddit_sentiment(query: str) -> Optional[float]:
    """
    Return average PRAW sentiment score for top Reddit posts.
    Returns None if credentials are missing.
    """
    if not (config.reddit_client_id and config.reddit_client_secret):
        return None
    try:
        import praw
        reddit = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent="prediction-bot/1.0",
        )
        scores = []
        for sub in reddit.subreddits.search(query, limit=3):
            for post in sub.hot(limit=10):
                score = _simple_lexicon_score(f"{post.title} {post.selftext}")
                scores.append(score)
        return statistics.mean(scores) if scores else None
    except Exception as e:
        logger.debug(f"Reddit sentiment failed: {e}")
        return None


# ── Main engine ───────────────────────────────────────────────────────────────

class SentimentEngine:
    """
    Aggregates news articles and social signals into a SentimentResult
    that downstream models can consume.
    """

    def __init__(self, scraper: Optional[NewsScraper] = None):
        self.scraper = scraper or NewsScraper()

    def analyse(self, question: str, articles: Optional[List[Article]] = None) -> SentimentResult:
        """
        Analyse sentiment for a market question.
        If *articles* is provided, skip fetching.
        """
        if articles is None:
            articles = self.scraper.fetch(question)

        if not articles:
            return SentimentResult(
                label="Neutral", score=0.0, confidence=0.0,
                article_count=0, bullish_count=0, bearish_count=0,
                neutral_count=0, summary="No articles found.",
            )

        # Score each article
        raw_scores = [_score_article(a) for a in articles]
        labels = [_classify_score(s) for s in raw_scores]

        bullish_count = labels.count("Bullish")
        bearish_count = labels.count("Bearish")
        neutral_count = labels.count("Neutral")

        # Weighted average (more recent articles get higher weight)
        weights = [1 / (i + 1) for i in range(len(raw_scores))]
        weighted_score = sum(s * w for s, w in zip(raw_scores, weights)) / sum(weights)

        # Blend with Reddit if available
        reddit_score = _fetch_reddit_sentiment(" ".join(question.split()[:5]))
        if reddit_score is not None:
            weighted_score = 0.7 * weighted_score + 0.3 * reddit_score
            logger.debug(f"Reddit score blended: {reddit_score:.3f}")

        label = _classify_score(weighted_score)

        # Confidence = proportion of articles agreeing with aggregate label
        agreement = max(bullish_count, bearish_count, neutral_count) / len(articles)
        confidence = round(agreement, 3)

        # Build summary
        summary = (
            f"Analysed {len(articles)} articles: "
            f"{bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral. "
            f"Aggregate: {label} (score={weighted_score:+.3f}, confidence={confidence:.0%})."
        )
        logger.info(f"Sentiment for '{question[:50]}': {label} ({weighted_score:+.3f})")

        return SentimentResult(
            label=label,
            score=round(weighted_score, 4),
            confidence=confidence,
            article_count=len(articles),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            summary=summary,
        )
