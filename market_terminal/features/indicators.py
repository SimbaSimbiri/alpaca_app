from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes OHLCV DataFrame so every indicator/backtest has same schema.
    """
    clean = df.copy()
    clean.columns = [str(col).lower().strip() for col in clean.columns]
    clean = clean.sort_index()

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in clean.columns]

    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    clean[required] = clean[required].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=required)

    return clean


def add_sma(df: pd.DataFrame, windows: tuple[int, ...] = (20, 50, 60, 200)) -> pd.DataFrame:
    out = df.copy()
    for window in windows:
        # sma for each window so we have added columns of size len(windows)
        out[f"sma_{window}"] = out["close"].rolling(window=window).mean()
    return out


def add_ema(df: pd.DataFrame, spans: tuple[int, ...] = (12, 20, 26, 50)) -> pd.DataFrame:
    out = df.copy()
    for span in spans:
        out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()
    return out


def add_macd(
    df: pd.DataFrame,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    out = df.copy()
    fast = out["close"].ewm(span=fast_span, adjust=False).mean()
    slow = out["close"].ewm(span=slow_span, adjust=False).mean()

    out["macd"] = fast - slow
    out["macd_signal"] = out["macd"].ewm(span=signal_span, adjust=False).mean()
    out["macd_histogram"] = out["macd"] - out["macd_signal"]

    return out


def add_momentum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    out = df.copy()
    out[f"momentum_{window}"] = out["close"] - out["close"].shift(window)
    out[f"momentum_pct_{window}"] = out["close"].pct_change(periods=window)
    return out


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index using Wilder-style exponential smoothing.
    RSI near 70 - overbought
    RSI near 30 - oversold.
    """
    out = df.copy()
    delta = out["close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Edge cases: if there are gains and no losses, RSI = 100.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    # If both gains and losses are zero, price is flat; 50 RSI.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    out[f"rsi_{window}"] = rsi
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    tr = true_range(out)
    out[f"atr_{window}"] = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return out


def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average Directional Index. ADX measures trend strength, not direction.
    A common trend-filter threshold is ADX > 25.
    """
    out = df.copy()

    up_move = out["high"].diff()
    down_move = -out["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=out.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=out.index,
    )

    atr = true_range(out).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    out[f"plus_di_{window}"] = plus_di
    out[f"minus_di_{window}"] = minus_di
    out[f"adx_{window}"] = adx

    return out


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    stdev_factor: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()
    middle = out["close"].rolling(window=window).mean()
    stdev = out["close"].rolling(window=window).std(ddof=0)

    out[f"bb_middle_{window}"] = middle
    out[f"bb_upper_{window}"] = middle + stdev_factor * stdev
    out[f"bb_lower_{window}"] = middle - stdev_factor * stdev
    out[f"bb_width_{window}"] = (out[f"bb_upper_{window}"] - out[f"bb_lower_{window}"]) / middle

    return out


def add_stochastic_oscillator(
    df: pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    out = df.copy()
    lowest_low = out["low"].rolling(window=k_window).min()
    highest_high = out["high"].rolling(window=k_window).max()
    denominator = (highest_high - lowest_low).replace(0, np.nan)

    out[f"stoch_k_{k_window}"] = 100 * (out["close"] - lowest_low) / denominator
    out[f"stoch_d_{d_window}"] = out[f"stoch_k_{k_window}"].rolling(window=d_window).mean()

    return out


def add_williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    highest_high = out["high"].rolling(window=window).max()
    lowest_low = out["low"].rolling(window=window).min()
    denominator = (highest_high - lowest_low).replace(0, np.nan)

    out[f"williams_r_{window}"] = -100 * (highest_high - out["close"]) / denominator
    return out


def add_obv(df: pd.DataFrame, smoothing_window: int = 20) -> pd.DataFrame:
    out = df.copy()
    direction = np.sign(out["close"].diff()).fillna(0)
    out["obv"] = (direction * out["volume"]).cumsum()
    out[f"obv_sma_{smoothing_window}"] = out["obv"].rolling(window=smoothing_window).mean()
    return out


def add_cmf(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Chaikin Money Flow. Positive values suggest accumulation / buying pressure;
    negative values suggest distribution / selling pressure.
    """
    out = df.copy()
    high_low_range = (out["high"] - out["low"]).replace(0, np.nan)
    money_flow_multiplier = ((out["close"] - out["low"]) - (out["high"] - out["close"])) / high_low_range
    money_flow_volume = money_flow_multiplier * out["volume"]

    out[f"cmf_{window}"] = money_flow_volume.rolling(window=window).sum() / out["volume"].rolling(window=window).sum()
    return out


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all assignment indicators. This intentionally includes more than six
    indicators so each strategy can draw from different categories.
    """
    out = prepare_ohlcv(df)
    out = add_sma(out)
    out = add_ema(out)
    out = add_macd(out)
    out = add_momentum(out)
    out = add_rsi(out)
    out = add_atr(out)
    out = add_adx(out)
    out = add_bollinger_bands(out)
    out = add_stochastic_oscillator(out)
    out = add_williams_r(out)
    out = add_obv(out)
    out = add_cmf(out)
    return out
