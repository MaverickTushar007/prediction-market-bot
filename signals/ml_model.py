"""
ML Signal Model — LightGBM + XGBoost ensemble for swing trading direction prediction.
Models are cached to disk and only retrained if data is >7 days old.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import pickle
import hashlib
import warnings
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
RETRAIN_DAYS = 7  # retrain every 7 days

@dataclass
class MLSignal:
    direction: str
    ml_probability: float
    ml_confidence: str
    feature_importance: dict
    model_agreement: float
    was_cached: bool = False


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    df["ret_1d"]  = close.pct_change(1)
    df["ret_3d"]  = close.pct_change(3)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-9)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"]     = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / (close + 1e-9)

    df["vol_ratio"]  = vol / (vol.rolling(20).mean() + 1e-9)
    df["vol_trend"]  = vol.rolling(5).mean() / (vol.rolling(20).mean() + 1e-9)
    df["dist_sma20"] = (close - sma20) / (sma20 + 1e-9)
    df["dist_sma50"] = (close - close.rolling(50).mean()) / (close.rolling(50).mean() + 1e-9)
    df["high_52w"]   = close / (close.rolling(252).max() + 1e-9)
    df["low_52w"]    = close / (close.rolling(252).min() + 1e-9)
    df["roc_5"]      = close.pct_change(5)
    df["roc_10"]     = close.pct_change(10)

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

FORWARD_DAYS  = 5
RETURN_THRESH = 0.02


def _model_path(symbol: str) -> Path:
    safe = symbol.replace("=","_").replace("^","_").replace("-","_")
    return MODELS_DIR / f"{safe}.pkl"


def _is_stale(model_path: Path) -> bool:
    if not model_path.exists():
        return True
    age_days = (datetime.now().timestamp() - model_path.stat().st_mtime) / 86400
    return age_days > RETRAIN_DAYS


def _save_model(symbol: str, ensemble: dict):
    path = _model_path(symbol)
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)


def _load_model(symbol: str) -> Optional[dict]:
    path = _model_path(symbol)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def fetch_training_data(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        tk   = yf.Ticker(symbol)
        hist = tk.history(period=period, interval="1d")
        if hist is None or len(hist) < 100:
            return None
        hist = hist.copy()
        hist = _compute_features(hist)
        hist["future_ret"] = hist["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
        hist["label"]      = (hist["future_ret"] > RETURN_THRESH).astype(int)
        hist.dropna(inplace=True)
        return hist
    except Exception as e:
        logger.warning(f"Training data failed for {symbol}: {e}")
        return None


def train_ensemble(symbol: str) -> Optional[dict]:
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        df = fetch_training_data(symbol)
        if df is None or len(df) < 80:
            return None

        X = df[FEATURE_COLS].values
        y = df["label"].values
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(np.unique(y_train)) < 2:
            return None

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:,1]
        xgb_auc  = roc_auc_score(y_test, xgb_prob) if len(np.unique(y_test)) > 1 else 0.5

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            feature_name=FEATURE_COLS,
        )
        lgb_prob = lgb_model.predict_proba(X_test)[:,1]
        lgb_auc  = roc_auc_score(y_test, lgb_prob) if len(np.unique(y_test)) > 1 else 0.5

        importance = dict(zip(FEATURE_COLS, xgb_model.feature_importances_))
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])

        ensemble = {
            "xgb": xgb_model,
            "lgb": lgb_model,
            "xgb_auc": round(xgb_auc, 3),
            "lgb_auc": round(lgb_auc, 3),
            "top_features": top_features,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_cols": FEATURE_COLS,
        }
        _save_model(symbol, ensemble)
        logger.info(f"Trained & saved model for {symbol} (AUC: XGB={xgb_auc:.3f} LGB={lgb_auc:.3f})")
        return ensemble

    except Exception as e:
        logger.warning(f"Training failed for {symbol}: {e}")
        return None


def predict(symbol: str, sentiment_score: float = 0.0) -> Optional[MLSignal]:
    try:
        path = _model_path(symbol)
        was_cached = False

        if _is_stale(path):
            logger.info(f"Training fresh model for {symbol}...")
            ensemble = train_ensemble(symbol)
        else:
            ensemble = _load_model(symbol)
            if ensemble is None:
                ensemble = train_ensemble(symbol)
            else:
                was_cached = True
                logger.info(f"Loaded cached model for {symbol}")

        if ensemble is None:
            return None

        # Get latest features
        df = fetch_training_data(symbol)
        if df is None or len(df) < 10:
            return None

        latest = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        latest_df = pd.DataFrame(latest, columns=FEATURE_COLS)

        xgb_prob = float(ensemble["xgb"].predict_proba(latest)[:,1])
        lgb_prob = float(ensemble["lgb"].predict_proba(latest_df)[:,1])

        sentiment_adj = (sentiment_score + 1) / 2
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
            was_cached=was_cached,
        )
    except Exception as e:
        logger.warning(f"ML prediction failed for {symbol}: {e}")
        return None
