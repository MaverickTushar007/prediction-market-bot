"""
ML Signal Model — LightGBM + XGBoost ensemble for swing trading direction prediction.
Trained on 1-year of historical OHLCV data per ticker, predicts 5-day forward return.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import pandas as pd
import yfinance as yf
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MLSignal:
    direction: str        # BUY / SELL
    ml_probability: float # 0-1, probability of BUY
    ml_confidence: str    # HIGH / MEDIUM / LOW
    feature_importance: dict  # top features driving prediction
    model_agreement: float    # how much XGB and LGBM agree


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20+ technical features from OHLCV data."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # Price momentum
    df["ret_1d"]  = close.pct_change(1)
    df["ret_3d"]  = close.pct_change(3)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20

    # ATR-14
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"]     = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close  # normalised ATR

    # Volume
    df["vol_ratio"]  = vol / vol.rolling(20).mean()
    df["vol_trend"]  = vol.rolling(5).mean() / vol.rolling(20).mean()

    # Price position
    df["dist_sma20"] = (close - sma20) / sma20
    df["dist_sma50"] = (close - close.rolling(50).mean()) / close.rolling(50).mean()
    df["high_52w"]   = close / close.rolling(252).max()
    df["low_52w"]    = close / close.rolling(252).min()

    # Momentum / rate of change
    df["roc_5"]  = close.pct_change(5)
    df["roc_10"] = close.pct_change(10)

    # Stochastic %K
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = (close - low14) / (high14 - low14 + 1e-9) * 100

    return df


FEATURE_COLS = [
    "ret_1d","ret_3d","ret_5d","ret_10d","ret_20d",
    "rsi","macd","macd_signal","macd_hist",
    "bb_pct","bb_width","atr_pct",
    "vol_ratio","vol_trend",
    "dist_sma20","dist_sma50","high_52w","low_52w",
    "roc_5","roc_10","stoch_k",
]

FORWARD_DAYS  = 5      # predict 5-day return
RETURN_THRESH = 0.02   # 2% threshold for BUY label


def fetch_training_data(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch historical data and engineer features + labels."""
    try:
        tk   = yf.Ticker(symbol)
        hist = tk.history(period=period, interval="1d")
        if hist is None or len(hist) < 100:
            return None
        hist = hist.copy()
        hist = _compute_features(hist)
        # Label: 1 if price rises >2% in next 5 days
        hist["future_ret"] = hist["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
        hist["label"]      = (hist["future_ret"] > RETURN_THRESH).astype(int)
        hist.dropna(inplace=True)
        return hist
    except Exception as e:
        logger.warning(f"Training data failed for {symbol}: {e}")
        return None


def train_ensemble(symbol: str) -> Optional[dict]:
    """Train XGBoost + LightGBM ensemble on historical data."""
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        df = fetch_training_data(symbol)
        if df is None or len(df) < 80:
            return None

        X = df[FEATURE_COLS].values
        y = df["label"].values

        # Time-series aware split (no lookahead)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(np.unique(y_train)) < 2:
            return None

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:,1]
        xgb_auc  = roc_auc_score(y_test, xgb_prob) if len(np.unique(y_test)) > 1 else 0.5

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        lgb_prob = lgb_model.predict_proba(X_test)[:,1]
        lgb_auc  = roc_auc_score(y_test, lgb_prob) if len(np.unique(y_test)) > 1 else 0.5

        # Feature importance from XGBoost
        importance = dict(zip(FEATURE_COLS, xgb_model.feature_importances_))
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])

        return {
            "xgb": xgb_model,
            "lgb": lgb_model,
            "xgb_auc": round(xgb_auc, 3),
            "lgb_auc": round(lgb_auc, 3),
            "top_features": top_features,
        }
    except Exception as e:
        logger.warning(f"Training failed for {symbol}: {e}")
        return None


def predict(symbol: str, sentiment_score: float = 0.0) -> Optional[MLSignal]:
    """Train model + predict on latest data point."""
    try:
        import xgboost as xgb
        import lightgbm as lgb

        df = fetch_training_data(symbol)
        if df is None or len(df) < 80:
            return None

        ensemble = train_ensemble(symbol)
        if ensemble is None:
            return None

        # Latest feature row
        latest = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)

        xgb_prob = float(ensemble["xgb"].predict_proba(latest)[0][1])
        lgb_prob = float(ensemble["lgb"].predict_proba(latest)[0][1])

        # Blend with sentiment score (10% weight)
        sentiment_adj = (sentiment_score + 1) / 2  # -1..1 → 0..1
        final_prob = 0.45 * xgb_prob + 0.45 * lgb_prob + 0.10 * sentiment_adj

        direction = "BUY" if final_prob > 0.52 else "SELL"
        agreement = 1.0 - abs(xgb_prob - lgb_prob)

        if final_prob > 0.65 or final_prob < 0.35:
            confidence = "HIGH"
        elif final_prob > 0.55 or final_prob < 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return MLSignal(
            direction=direction,
            ml_probability=round(final_prob, 3),
            ml_confidence=confidence,
            feature_importance=ensemble["top_features"],
            model_agreement=round(agreement, 3),
        )
    except Exception as e:
        logger.warning(f"ML prediction failed for {symbol}: {e}")
        return None
