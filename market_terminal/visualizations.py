from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from market_terminal.backtester import BacktestResult
from market_terminal.performance import drawdown_series


def _trade_points(trades: pd.DataFrame, action: str) -> pd.DataFrame:
    if trades.empty:
        return trades

    return trades[trades["action"].str.upper() == action.upper()]


def plot_strategy_price_chart(
    df: pd.DataFrame,
    result: BacktestResult,
    output_path: Path,
) -> Path:
    """
    Price chart with indicators and executed buy/sell markers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df["close"], label="Close")

    name = result.name.lower()

    if "trend" in name:
        for col in ["sma_50", "sma_200"]:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col.upper())

    elif "mean" in name:
        for col in ["bb_middle_20", "bb_upper_20", "bb_lower_20"]:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col.replace("_", " ").title())

    elif "custom" in name:
        for col in ["sma_50", "sma_200", "bb_middle_20"]:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col.replace("_", " ").title())

    buys = _trade_points(result.trades, "BUY")
    sells = _trade_points(result.trades, "SELL")

    if not buys.empty:
        ax.scatter(buys.index, buys["price"], marker="^", s=80, label="Buy", zorder=5)

    if not sells.empty:
        ax.scatter(sells.index, sells["price"], marker="v", s=80, label="Sell", zorder=5)

    ax.set_title(f"{result.name}: Price, Indicators, and Executed Trades")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_equity_curves(
    results: list[BacktestResult],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    for result in results:
        ax.plot(result.portfolio.index, result.portfolio["equity"], label=result.name)

    ax.set_title("Equity Curve Comparison")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_drawdown_curves(
    results: list[BacktestResult],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    for result in results:
        dd = drawdown_series(result.portfolio["equity"])
        ax.plot(dd.index, dd, label=result.name)

    ax.set_title("Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def save_all_charts(
    df: pd.DataFrame,
    results: list[BacktestResult],
    chart_dir: Path,
) -> list[Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for result in results:
        if result.name == "Buy & Hold":
            continue

        safe_name = (
            result.name.lower()
            .replace("&", "and")
            .replace(":", "")
            .replace(" ", "_")
            .replace("/", "_")
        )
        paths.append(plot_strategy_price_chart(df, result, chart_dir / f"{safe_name}_price_signals.png"))

    paths.append(plot_equity_curves(results, chart_dir / "equity_curve_comparison.png"))
    paths.append(plot_drawdown_curves(results, chart_dir / "drawdown_comparison.png"))

    return paths
