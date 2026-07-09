from __future__ import annotations

import numpy as np
import pandas as pd

from market_terminal.features.indicators import add_all_indicators, prepare_ohlcv


FEATURE_COLUMNS = [
    # Returns / rolling statistics
    "log_return",
    "rolling_mean_10",
    "rolling_std_10",

    # Trend features
    "sma_20_dist",
    "sma_50_dist",
    "sma_200_dist",
    "ema_20_dist",
    "ema_50_dist",
    "macd",
    "macd_signal",
    "macd_histogram",
    "adx_14",

    # Momentum features
    "momentum_pct_10",
    "rsi_14",
    "stoch_k_14",
    "stoch_d_3",
    "williams_r_14",

    # Volatility features
    "atr_14_pct",
    "bb_width_20",

    # Volume features
    "obv_change",
    "obv_sma_ratio",
    "cmf_20",
]


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full machine-learning feature dataset.

    Output includes:
    - ML-normalized feature columns
    - Binary next-day target
    """

    out = prepare_ohlcv(df)
    out = add_all_indicators(out)

    # Basic return features
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["rolling_mean_10"] = out["log_return"].rolling(window=10).mean()
    out["rolling_std_10"] = out["log_return"].rolling(window=10).std()

    # Normalize moving averages as distance from close.
    # This is usually better for ML than raw SMA/EMA values because it removes price scale.
    out["sma_20_dist"] = out["close"] / out["sma_20"] - 1
    out["sma_50_dist"] = out["close"] / out["sma_50"] - 1
    out["sma_200_dist"] = out["close"] / out["sma_200"] - 1

    out["ema_20_dist"] = out["close"] / out["ema_20"] - 1
    out["ema_50_dist"] = out["close"] / out["ema_50"] - 1

    # Normalize ATR by price.
    # Raw ATR is dollar-based, so ATR / close makes it comparable across price regimes.
    out["atr_14_pct"] = out["atr_14"] / out["close"]

    # OBV can become extremely large because it is cumulative.
    # Use daily percentage change plus OBV/SMA ratio instead.
    out["obv_change"] = out["obv"].pct_change()
    out["obv_sma_ratio"] = out["obv"] / out["obv_sma_20"].replace(0, np.nan) - 1

    # Target: 1 if next-day return is positive, else 0.
    out["next_day_return"] = out["close"].shift(-1) / out["close"] - 1
    out["target"] = np.where(out["next_day_return"] > 0, 1, 0)

    # Last row has no known next-day return.
    out.loc[out["next_day_return"].isna(), "target"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def get_clean_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean ML-ready DataFrame containing all feature columns and target.
    """

    out = add_ml_features(df)

    required_columns = FEATURE_COLUMNS + ["target"]
    out = out.dropna(subset=required_columns).copy()

    out["target"] = out["target"].astype(int)

    return out