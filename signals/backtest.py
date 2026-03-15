"""
Backtest engine — evaluates ML signal accuracy on historical data.
For each ticker, walks through past 6 months day by day,
generates a signal, then checks if it was correct 5 days later.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

FORWARD_DAYS  = 5
RETURN_THRESH = 0.02  # 2% threshold


def backtest_ticker(symbol: str, lookback_days: int = 180) -> Optional[dict]:
    """
    Run backtest for a single ticker.
    Returns win rate, avg return, Sharpe, max drawdown.
    """
    try:
        from signals.ml_model import _compute_features, FEATURE_COLS, _load_model, train_ensemble, _is_stale, _model_path

        # Fetch data
        tk = yf.Ticker(symbol)
        hist = tk.history(period="2y", interval="1d")
        if hist is None or len(hist) < 150:
            return None

        hist = _compute_features(hist.copy())
        hist["future_ret"] = hist["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
        hist["label"]      = (hist["future_ret"] > RETURN_THRESH).astype(int)
        hist.dropna(inplace=True)

        if len(hist) < 100:
            return None

        # Load or train model
        path = _model_path(symbol)
        if _is_stale(path):
            ensemble = train_ensemble(symbol)
        else:
            ensemble = _load_model(symbol)
        if ensemble is None:
            return None

        # Walk-forward backtest on last lookback_days
        test_start = max(0, len(hist) - lookback_days - FORWARD_DAYS)
        test_df    = hist.iloc[test_start:-FORWARD_DAYS]  # exclude last 5 rows (no future data)

        if len(test_df) < 20:
            return None

        X = test_df[FEATURE_COLS].values
        xgb_probs = ensemble["xgb"].predict_proba(X)[:,1]
        lgb_probs = ensemble["lgb"].predict_proba(
            pd.DataFrame(X, columns=FEATURE_COLS)
        )[:,1]
        final_probs = 0.5 * xgb_probs + 0.5 * lgb_probs

        directions = np.where(final_probs > 0.52, "BUY", "SELL")
        actual_rets = test_df["future_ret"].values

        # Calculate trade returns
        trade_rets = []
        for i, (direction, actual_ret) in enumerate(zip(directions, actual_rets)):
            if direction == "BUY":
                trade_rets.append(actual_ret)
            else:  # SELL
                trade_rets.append(-actual_ret)

        trade_rets = np.array(trade_rets)

        # Only count confident signals
        confident_mask = (final_probs > 0.58) | (final_probs < 0.42)
        confident_rets = trade_rets[confident_mask]

        if len(confident_rets) == 0:
            confident_rets = trade_rets

        wins       = np.sum(confident_rets > 0)
        total      = len(confident_rets)
        win_rate   = wins / total if total > 0 else 0
        avg_ret    = np.mean(confident_rets)
        std_ret    = np.std(confident_rets)
        sharpe     = (avg_ret / std_ret * np.sqrt(252/FORWARD_DAYS)) if std_ret > 0 else 0

        # Cumulative returns for drawdown
        cum_rets   = np.cumprod(1 + confident_rets) - 1
        peak       = np.maximum.accumulate(cum_rets + 1)
        drawdown   = (cum_rets + 1 - peak) / peak
        max_dd     = float(np.min(drawdown))

        # Monthly breakdown (last 3 months)
        monthly = []
        dates = test_df.index[-len(confident_rets):]
        for month_offset in range(3):
            pass  # simplified

        return {
            "symbol":    symbol,
            "win_rate":  round(float(win_rate) * 100, 1),
            "avg_ret":   round(float(avg_ret) * 100, 2),
            "sharpe":    round(float(sharpe), 2),
            "max_dd":    round(float(max_dd) * 100, 1),
            "n_trades":  int(total),
            "total_ret": round(float(np.sum(confident_rets) * 100), 1),
            "best_trade": round(float(np.max(confident_rets) * 100), 1),
            "worst_trade": round(float(np.min(confident_rets) * 100), 1),
        }

    except Exception as e:
        logger.warning(f"Backtest failed for {symbol}: {e}")
        return None


def run_backtest(symbols: list, max_tickers: int = 20) -> list:
    """Run backtest across multiple tickers."""
    results = []
    for sym in symbols[:max_tickers]:
        r = backtest_ticker(sym)
        if r:
            results.append(r)
            logger.info(f"Backtest {sym}: WR={r['win_rate']}% Sharpe={r['sharpe']}")
    return results
