"""
News Scraper — collects recent news articles relevant to a market question
via RSS feeds and the NewsAPI (with mock fallback).
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from utils.config import config
from utils.helpers import get_logger, retry

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Article:
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    content: str = ""

    @property
    def text(self) -> str:
        return f"{self.title}. {self.description} {self.content}".strip()

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["published_at"] = self.published_at.isoformat()
        return d


# ── Mock article templates ─────────────────────────────────────────────────────

_BULLISH_SNIPPETS = [
    "Analysts predict strong performance",
    "Positive indicators suggest likelihood increasing",
    "Market consensus shifting bullish",
    "Recent data supports affirmative outcome",
    "Experts now lean toward yes",
    "Momentum building toward expected outcome",
]
_BEARISH_SNIPPETS = [
    "Analysts remain skeptical",
    "Data points toward negative outcome",
    "Growing consensus against the event",
    "Recent headwinds make outcome less likely",
    "Experts downgrading probability",
    "Market participants pricing in failure",
]
_NEUTRAL_SNIPPETS = [
    "Outcome remains highly uncertain",
    "Divided opinion among analysts",
    "Mixed signals from recent data",
    "Both scenarios considered plausible",
    "No clear consensus emerging",
]

_SOURCES = ["Reuters", "Bloomberg", "AP News", "The Guardian", "BBC", "FT", "WSJ"]


def _mock_articles_for(question: str, n: int = 5) -> List[Article]:
    """Generate plausible-sounding mock news articles for a question."""
    seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    articles: List[Article] = []

    for i in range(n):
        bias = rng.choice(["bullish", "bullish", "bearish", "neutral"])
        snippet = rng.choice(
            _BULLISH_SNIPPETS if bias == "bullish" else
            _BEARISH_SNIPPETS if bias == "bearish" else
            _NEUTRAL_SNIPPETS
        )
        short_q = question[:60].rstrip("?")
        title = f"{snippet}: {short_q}"
        description = (
            f"In a detailed analysis, financial reporters note that {short_q.lower()} "
            f"has generated significant commentary. {snippet}. "
            f"Stakeholders are watching closely as the deadline approaches."
        )
        articles.append(Article(
            title=title,
            description=description,
            url=f"https://mock-news.example.com/article-{i}",
            source=rng.choice(_SOURCES),
            published_at=now - timedelta(hours=rng.randint(1, 72)),
            content=f"Extended coverage: {description} Multiple analysts weighed in.",
        ))

    return articles


# ── RSS Fetcher ────────────────────────────────────────────────────────────────

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.cnn.com/rss/money_news_international.rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]


def _keywords_from_question(question: str) -> List[str]:
    """Extract search keywords from a market question."""
    import re
    stop_words = {"will", "the", "a", "an", "in", "of", "by", "is", "be",
                  "to", "for", "and", "or", "at", "from", "with", "than",
                  "that", "this", "before", "after", "qualify", "announce"}
    # Keep numbers, acronyms, proper nouns
    text = re.sub(r"[?]", "", question)
    words = text.split()
    keywords = [w.strip(".,") for w in words
                if w.lower().strip(".,") not in stop_words and len(w) > 2]
    return keywords[:6]


def _article_matches(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


@retry(max_attempts=2)
def _fetch_rss(url: str, keywords: List[str], max_items: int = 10) -> List[Article]:
    """Fetch and filter articles from an RSS feed."""
    try:
        import feedparser
    except ImportError:
        return []

    feed = feedparser.parse(url)
    articles: List[Article] = []
    for entry in feed.entries[:50]:
        text = f"{entry.get('title', '')} {entry.get('summary', '')}"
        if not _article_matches(text, keywords):
            continue
        try:
            pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pub = datetime.now(timezone.utc)
        articles.append(Article(
            title=entry.get("title", ""),
            description=entry.get("summary", ""),
            url=entry.get("link", ""),
            source=feed.feed.get("title", url),
            published_at=pub,
        ))
        if len(articles) >= max_items:
            break
    return articles


# ── NewsAPI client ─────────────────────────────────────────────────────────────

@retry(max_attempts=2)
def _fetch_newsapi(query: str, api_key: str) -> List[Article]:
    """Fetch top relevant articles from NewsAPI."""
    if not api_key:
        return []
    try:
        import requests
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "relevancy",
                "pageSize": 10,
                "language": "en",
                "apiKey": api_key,
            },
            timeout=8,
        )
        resp.raise_for_status()
        articles = []
        for a in resp.json().get("articles", []):
            articles.append(Article(
                title=a.get("title", ""),
                description=a.get("description", "") or "",
                url=a.get("url", ""),
                source=a.get("source", {}).get("name", ""),
                published_at=datetime.fromisoformat(
                    a["publishedAt"].replace("Z", "+00:00")
                ),
                content=a.get("content", "") or "",
            ))
        return articles
    except Exception as e:
        logger.warning(f"NewsAPI error: {e}")
        return []


# ── Main scraper class ─────────────────────────────────────────────────────────

class NewsScraper:
    """
    Gathers recent news for a market question from multiple sources.
    Falls back to mock data if no API keys are configured.
    """

    def __init__(
        self,
        news_api_key: str = config.news_api_key,
        use_rss: bool = True,
    ):
        self.news_api_key = news_api_key
        self.use_rss = use_rss

    def fetch(self, question: str, max_articles: int = 15) -> List[Article]:
        """Return up to *max_articles* relevant articles for *question*."""
        keywords = _keywords_from_question(question)
        query = " ".join(keywords[:3])
        articles: List[Article] = []

        # 1. NewsAPI
        na_articles = _fetch_newsapi(query, self.news_api_key)
        articles.extend(na_articles)

        # 2. RSS
        if self.use_rss:
            for feed_url in RSS_FEEDS:
                try:
                    rss_arts = _fetch_rss(feed_url, keywords, max_items=5)
                    articles.extend(rss_arts)
                    if len(articles) >= max_articles:
                        break
                except Exception:
                    continue

        # 3. Mock fallback
        if not articles:
            logger.info(f"No live news found for '{question[:40]}' — using mocks.")
            articles = _mock_articles_for(question, n=6)

        # Deduplicate by title, sort newest first
        seen: set[str] = set()
        unique: List[Article] = []
        for a in sorted(articles, key=lambda x: x.published_at, reverse=True):
            if a.title not in seen:
                seen.add(a.title)
                unique.append(a)

        return unique[:max_articles]
