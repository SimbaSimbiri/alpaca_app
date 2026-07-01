from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    name: str
    signal_column: str
    description: str
    entry_rules: list[str]
    exit_rules: list[str]


def build_stateful_long_only_position(
    entry_signal: pd.Series,
    exit_signal: pd.Series,
) -> pd.Series:
    """
    Converts entry/exit booleans into a long-only target position series.

    1 means the strategy wants to be long.
    0 means the strategy wants to be in cash.
    """
    entry = entry_signal.fillna(False).astype(bool)
    exit_ = exit_signal.fillna(False).astype(bool)

    in_position = False
    positions: list[int] = []

    for timestamp in entry.index:
        if in_position and exit_.loc[timestamp]:
            in_position = False
        elif not in_position and entry.loc[timestamp]:
            in_position = True

        positions.append(1 if in_position else 0)

    return pd.Series(positions, index=entry.index, name="target_position")


def add_trend_following_strategy(df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySpec]:
    """
    Strategy 1: trend-following.

    Uses trend indicators: MACD, ADX, and moving averages.
    """
    out = df.copy()

    entry = (
        (out["macd"] > out["macd_signal"])
        & (out["adx_14"] > 25)
        & (out["sma_50"] > out["sma_200"])
        & (out["close"] > out["sma_50"])
    )

    exit_ = (
        (out["macd"] < out["macd_signal"])
        | (out["close"] < out["sma_50"])
        | (out["adx_14"] < 18)
    )

    out["trend_following_signal"] = build_stateful_long_only_position(entry, exit_)

    spec = StrategySpec(
        name="Trend Following",
        signal_column="trend_following_signal",
        description="Long-only trend system using MACD direction, ADX trend strength, and moving-average trend filters.",
        entry_rules=[
            "MACD is above its signal line.",
            "ADX(14) is above 25, implying a stronger trend regime.",
            "SMA(50) is above SMA(200), implying an intermediate/long-term uptrend.",
            "Close is above SMA(50).",
        ],
        exit_rules=[
            "MACD crosses below its signal line, or",
            "Close falls below SMA(50), or",
            "ADX(14) falls below 18, implying trend strength has faded.",
        ],
    )

    return out, spec


def add_mean_reversion_strategy(df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySpec]:
    """
    Strategy 2: mean reversion.

    Uses momentum/oscillator and volatility indicators: RSI and Bollinger Bands.
    """
    out = df.copy()

    entry = (
        (out["rsi_14"] < 30)
        & (out["close"] < out["bb_lower_20"])
    )

    # The first exit is the assignment's classic overbought exit. The middle-band
    # exit is a practical risk-management exit so the strategy does not hold a
    # mean-reversion trade indefinitely waiting for RSI > 70.
    exit_ = (
        ((out["rsi_14"] > 70) & (out["close"] > out["bb_upper_20"]))
        | (out["close"] > out["bb_middle_20"])
        | (out["rsi_14"] > 55)
    )

    out["mean_reversion_signal"] = build_stateful_long_only_position(entry, exit_)

    spec = StrategySpec(
        name="Mean Reversion",
        signal_column="mean_reversion_signal",
        description="Long-only oversold bounce system using RSI and Bollinger Bands.",
        entry_rules=[
            "RSI(14) is below 30.",
            "Close is below the lower Bollinger Band(20, 2).",
        ],
        exit_rules=[
            "RSI(14) is above 70 and close is above the upper Bollinger Band, or",
            "Close reverts above the Bollinger middle band, or",
            "RSI(14) rises above 55 as a faster neutral-exit rule.",
        ],
    )

    return out, spec


def add_custom_strategy(df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySpec]:
    """
    Strategy 3: custom multi-category system.

    Combines indicators from trend, momentum, volatility, and volume categories.
    The idea is to participate only when the broad trend is positive, price has
    reclaimed the Bollinger middle band, momentum is positive, and volume flow
    confirms accumulation.
    """
    out = df.copy()

    entry = (
        (out["close"] > out["sma_200"])
        & (out["momentum_10"] > 0)
        & (out["close"] > out["bb_middle_20"])
        & (out["obv"] > out["obv_sma_20"])
        & (out["cmf_20"] > 0)
    )

    exit_ = (
        (out["close"] < out["sma_50"])
        | (out["momentum_10"] < 0)
        | (out["cmf_20"] < 0)
        | (out["rsi_14"] > 75)
    )

    out["custom_strategy_signal"] = build_stateful_long_only_position(entry, exit_)

    spec = StrategySpec(
        name="Custom: Volume-Confirmed Trend Pullback",
        signal_column="custom_strategy_signal",
        description=(
            "Long-only custom strategy combining trend, momentum, volatility, and volume. "
            "It enters when price is in a broad uptrend, momentum is positive, price has "
            "reclaimed the Bollinger middle band, and OBV/CMF confirm accumulation."
        ),
        entry_rules=[
            "Trend: close is above SMA(200).",
            "Momentum: Momentum(10) is positive.",
            "Volatility/location: close is above the Bollinger middle band.",
            "Volume: OBV is above OBV SMA(20).",
            "Volume: CMF(20) is above zero.",
        ],
        exit_rules=[
            "Close falls below SMA(50), or",
            "Momentum(10) turns negative, or",
            "CMF(20) turns negative, or",
            "RSI(14) exceeds 75, indicating an overheated move.",
        ],
    )

    return out, spec


def add_all_strategy_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[StrategySpec]]:
    out, trend_spec = add_trend_following_strategy(df)
    out, mean_spec = add_mean_reversion_strategy(out)
    out, custom_spec = add_custom_strategy(out)

    return out, [trend_spec, mean_spec, custom_spec]
