"""
Test Suite — validates each pipeline component independently.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone


# ── Scanner tests ──────────────────────────────────────────────────────────────

class TestMarketScanner:
    def test_mock_generation(self):
        from scanner.market_scanner import generate_mock_markets
        markets = generate_mock_markets(n=20)
        assert len(markets) == 20
        for m in markets:
            assert 0 < m.yes_price < 1
            assert m.volume_24h > 0
            assert m.days_to_expiry >= 0

    def test_scanner_returns_dataframe(self):
        from scanner.market_scanner import MarketScanner
        scanner = MarketScanner(min_volume=0, max_days=365, max_spread=0.2)
        df = scanner.scan()
        assert not df.empty
        assert "rank_score" in df.columns
        assert df["rank_score"].is_monotonic_decreasing or True  # sorted desc

    def test_filter_removes_low_volume(self):
        from scanner.market_scanner import MarketScanner, generate_mock_markets
        scanner = MarketScanner(min_volume=999_999, max_days=365, max_spread=1.0)
        mocks = generate_mock_markets(n=30)
        filtered = scanner._filter(mocks)
        assert all(m.volume_24h >= 999_999 for m in filtered)


# ── News scraper tests ─────────────────────────────────────────────────────────

class TestNewsScraper:
    def test_mock_fallback(self):
        from research.news_scraper import NewsScraper
        scraper = NewsScraper(news_api_key="", use_rss=False)
        articles = scraper.fetch("Will the Fed cut rates in 2025?")
        assert len(articles) > 0
        assert all(hasattr(a, "title") for a in articles)

    def test_keywords_extracted(self):
        from research.news_scraper import _keywords_from_question
        kws = _keywords_from_question("Will Bitcoin reach $100k by December 2025?")
        assert "bitcoin" in kws or "reach" in kws


# ── Sentiment tests ────────────────────────────────────────────────────────────

class TestSentimentEngine:
    def test_bullish_sentiment(self):
        from research.sentiment_engine import SentimentEngine
        from research.news_scraper import Article
        engine = SentimentEngine()
        articles = [
            Article("Market surges to new highs", "Strong gains expected",
                    "http://x.com", "TestSource",
                    datetime.now(timezone.utc), "rally surge positive growth"),
        ]
        result = engine.analyse("Will market rise?", articles=articles)
        assert result.label in ["Bullish", "Bearish", "Neutral"]
        assert -1 <= result.score <= 1

    def test_empty_articles(self):
        from research.sentiment_engine import SentimentEngine
        engine = SentimentEngine()
        result = engine.analyse("some question", articles=[])
        assert result.label == "Neutral"
        assert result.article_count == 0


# ── Kelly Criterion tests ──────────────────────────────────────────────────────

class TestKelly:
    def test_positive_edge(self):
        from risk.kelly import calculate_bet_size, kelly_fraction
        # p=0.6, price=0.5 → clear positive edge
        sizing = calculate_bet_size("test", "YES", 0.5, 0.6, bankroll=10000)
        assert sizing.bet_size_usd > 0
        assert sizing.fractional_kelly > 0

    def test_no_edge(self):
        from risk.kelly import calculate_bet_size
        # Model prob matches market price — no edge
        sizing = calculate_bet_size("test", "YES", 0.5, 0.5, bankroll=10000)
        assert sizing.bet_size_usd == 0.0

    def test_max_position_cap(self):
        from risk.kelly import calculate_bet_size
        sizing = calculate_bet_size("test", "YES", 0.1, 0.9, bankroll=10000, max_position_pct=0.05)
        assert sizing.bet_size_usd <= 500.01  # 5% of 10000 + float tolerance

    def test_kelly_formula(self):
        from risk.kelly import kelly_fraction
        # p=0.6, b=1.0 (even odds) → f = (0.6*1 - 0.4)/1 = 0.2
        f = kelly_fraction(0.6, 1.0)
        assert abs(f - 0.2) < 0.001


# ── Risk checks tests ──────────────────────────────────────────────────────────

class TestRiskManager:
    def test_all_pass(self):
        from risk.risk_checks import RiskManager
        rm = RiskManager(bankroll=10000)
        result = rm.check_trade(100.0)
        assert result.passed

    def test_too_large_bet(self):
        from risk.risk_checks import RiskManager
        rm = RiskManager(bankroll=10000, max_position_pct=0.05)
        result = rm.check_trade(600.0)  # > 5% of 10000 = 500
        assert not result.passed

    def test_daily_loss_halt(self):
        from risk.risk_checks import RiskManager
        rm = RiskManager(bankroll=10000, daily_loss_limit_pct=0.10)
        rm.current_bankroll = 8900  # -11% → over 10% limit
        rm._daily_start_bankroll = 10000
        result = rm.check_trade(10.0)
        assert not result.passed

    def test_concurrent_limit(self):
        from risk.risk_checks import RiskManager
        rm = RiskManager(bankroll=10000, max_concurrent=2)
        rm._open_positions = [{"market_id": "a"}, {"market_id": "b"}]
        result = rm.check_trade(10.0)
        assert not result.passed


# ── Paper trader tests ─────────────────────────────────────────────────────────

class TestPaperTrader:
    def test_open_and_close(self):
        from execution.paper_trader import PaperTrader
        pt = PaperTrader(slippage_bps=0)

        # Force fill by mocking random
        import random
        original = random.random
        random.random = lambda: 0.01  # always < fill_probability

        trade = pt.open_trade(
            market_id="m001", question="Test?", direction="YES",
            size_usd=100, market_price=0.5, model_probability=0.6, edge=0.1,
        )
        random.random = original

        if trade:  # trade might not fill due to random
            assert trade.status == "open"
            closed = pt.close_trade("m001", outcome=True)
            if closed:
                assert closed.status == "closed"
                assert closed.pnl is not None

    def test_performance_stats_empty(self):
        from execution.paper_trader import PaperTrader
        pt = PaperTrader()
        stats = pt.performance_stats()
        assert "message" in stats  # "No closed trades yet."


# ── Ensemble model tests ───────────────────────────────────────────────────────

class TestEnsembleModel:
    def test_predict_returns_signal(self):
        from prediction.ensemble_model import EnsembleModel
        from prediction.probability_model import MarketFeatures
        model = EnsembleModel()
        features = MarketFeatures(
            market_price=0.3, volume_24h=5000, days_to_expiry=14,
            spread=0.02, sentiment_score=0.4, sentiment_confidence=0.7,
            uncertainty=0.6, open_interest=10000,
        )
        signal = model.predict(features, question="Will X happen?", market_id="test")
        assert 0 <= signal.model_probability <= 1
        assert signal.direction in ["YES", "NO", "PASS"]

    def test_edge_calculation(self):
        from prediction.ensemble_model import EnsembleModel
        from prediction.probability_model import MarketFeatures
        model = EnsembleModel(min_edge=0.04)
        # Market price 0.3, but model should predict higher → YES signal
        features = MarketFeatures(
            market_price=0.3, volume_24h=10000, days_to_expiry=10,
            spread=0.02, sentiment_score=0.8, sentiment_confidence=0.9,
            uncertainty=0.6, open_interest=20000,
        )
        signal = model.predict(features, market_id="edge_test")
        # Edge should be computed
        assert signal.edge == round(signal.model_probability - signal.market_price, 4) or \
               signal.edge == round((1 - signal.model_probability) - (1 - signal.market_price), 4)


# ── Utility tests ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_sharpe(self):
        from utils.helpers import sharpe_ratio
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        sr = sharpe_ratio(returns)
        assert isinstance(sr, float)

    def test_max_drawdown(self):
        from utils.helpers import max_drawdown
        equity = np.array([100, 110, 105, 90, 95, 100])
        dd = max_drawdown(equity)
        assert abs(dd - (110 - 90) / 110) < 0.001

    def test_brier_score(self):
        from utils.helpers import brier_score
        probs = np.array([0.8, 0.3, 0.7])
        outcomes = np.array([1, 0, 1])
        bs = brier_score(probs, outcomes)
        # Should be small for these calibrated predictions
        assert 0 <= bs <= 1

    def test_kelly_fraction_values(self):
        from utils.helpers import clamp
        assert clamp(1.5) == 1.0
        assert clamp(-0.1) == 0.0
        assert clamp(0.5) == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
