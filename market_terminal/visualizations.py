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

    elif "ml" in name:
        if "ml_signal" in df.columns:
            long_days = df[df["ml_signal"] == 1]

            if not long_days.empty:
                ax.scatter(
                    long_days.index,
                    long_days["close"],
                    marker="^",
                    s=80,
                    label="ML Long Signal",
                    zorder=5,
                )

    buys = _trade_points(result.trades, "BUY")
    sells = _trade_points(result.trades, "SELL")

    if not buys.empty:
        ax.scatter(
            buys.index,
            buys["price"],
            marker="^",
            s=80,
            label="Executed Buy",
            zorder=5,
        )

    if not sells.empty:
        ax.scatter(
            sells.index,
            sells["price"],
            marker="v",
            s=80,
            label="Executed Sell",
            zorder=5,
        )

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
        ax.plot(
            result.portfolio.index,
            result.portfolio["equity"],
            label=result.name,
        )

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

        paths.append(
            plot_strategy_price_chart(
                df=df,
                result=result,
                output_path=chart_dir / f"{safe_name}_price_signals.png",
            )
        )

    paths.append(
        plot_equity_curves(
            results=results,
            output_path=chart_dir / "equity_curve_comparison.png",
        )
    )

    paths.append(
        plot_drawdown_curves(
            results=results,
            output_path=chart_dir / "drawdown_comparison.png",
        )
    )

    return paths


# -------------------------------------------------------------------
# HW3 visualization helpers
# -------------------------------------------------------------------


def plot_hw3_equity_curve(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Plots Buy & Hold equity versus ML Signal equity.

    Required columns:
    - portfolio_value
    - buy_hold_value
    """

    required_columns = {"portfolio_value", "buy_hold_value"}
    missing = required_columns.difference(comparison_df.columns)

    if missing:
        raise ValueError(f"Missing columns for equity curve plot: {missing}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        comparison_df.index,
        comparison_df["buy_hold_value"],
        label="Buy & Hold",
    )

    ax.plot(
        comparison_df.index,
        comparison_df["portfolio_value"],
        label="ML Signal",
    )

    ax.set_title("AAPL Equity Curve: Buy & Hold vs ML Signal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_hw3_drawdown(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Plots Buy & Hold drawdown versus ML Signal drawdown.

    Required columns:
    - drawdown
    - buy_hold_drawdown
    """

    required_columns = {"drawdown", "buy_hold_drawdown"}
    missing = required_columns.difference(comparison_df.columns)

    if missing:
        raise ValueError(f"Missing columns for drawdown plot: {missing}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        comparison_df.index,
        comparison_df["buy_hold_drawdown"],
        label="Buy & Hold Drawdown",
    )

    ax.plot(
        comparison_df.index,
        comparison_df["drawdown"],
        label="ML Signal Drawdown",
    )

    ax.set_title("AAPL Drawdown: Buy & Hold vs ML Signal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_hw3_pca_variance(
    fitted_pca,
    output_path: str | Path,
) -> Path:
    """
    Plots individual and cumulative explained variance for selected PCA components.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    explained = fitted_pca.explained_variance_ratio
    cumulative = fitted_pca.cumulative_variance_ratio
    component_numbers = range(1, len(explained) + 1)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        component_numbers,
        explained,
        label="Individual Explained Variance",
    )

    ax.plot(
        component_numbers,
        cumulative,
        marker="o",
        label="Cumulative Explained Variance",
    )

    ax.axhline(
        0.80,
        linestyle="--",
        label="80% Threshold",
    )

    ax.set_title("AAPL PCA Explained Variance")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_hw3_ml_signals(
    test_data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Plots AAPL close price and marks days where the ML signal is long.

    Required columns:
    - close
    - ml_signal
    """

    required_columns = {"close", "ml_signal"}
    missing = required_columns.difference(test_data.columns)

    if missing:
        raise ValueError(f"Missing columns for ML signal plot: {missing}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    long_days = test_data[test_data["ml_signal"] == 1]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        test_data.index,
        test_data["close"],
        label="AAPL Close",
    )

    if not long_days.empty:
        ax.scatter(
            long_days.index,
            long_days["close"],
            marker="^",
            s=80,
            label="ML Long Signal",
            zorder=5,
        )

    ax.set_title("AAPL Close Price with ML Long Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_hw3_probability_signal(
    test_data: pd.DataFrame,
    output_path: str | Path,
    threshold: float = 0.60,
) -> Path:
    """
    Plots the model's predicted probability and the long-entry threshold.

    Required columns:
    - ml_probability
    """

    required_columns = {"ml_probability"}
    missing = required_columns.difference(test_data.columns)

    if missing:
        raise ValueError(f"Missing columns for probability plot: {missing}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        test_data.index,
        test_data["ml_probability"],
        label="Predicted Probability of Positive Next-Day Return",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        label=f"Long Threshold = {threshold:.2f}",
    )

    ax.set_title("AAPL ML Probability Signal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def save_hw3_charts(
    test_data: pd.DataFrame,
    comparison_df: pd.DataFrame,
    fitted_pca,
    chart_dir: str | Path,
) -> list[Path]:
    """
    Saves all HW3-specific charts.

    Charts:
    - equity curve comparison
    - drawdown comparison
    - PCA explained variance
    - close price with ML long signals
    - ML probability signal
    """

    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_hw3_equity_curve(
            comparison_df=comparison_df,
            output_path=chart_dir / "hw3_aapl_equity_curve.png",
        ),
        plot_hw3_drawdown(
            comparison_df=comparison_df,
            output_path=chart_dir / "hw3_aapl_drawdown.png",
        ),
        plot_hw3_pca_variance(
            fitted_pca=fitted_pca,
            output_path=chart_dir / "hw3_aapl_pca_variance.png",
        ),
        plot_hw3_ml_signals(
            test_data=test_data,
            output_path=chart_dir / "hw3_aapl_ml_signals.png",
        ),
        plot_hw3_probability_signal(
            test_data=test_data,
            output_path=chart_dir / "hw3_aapl_probability_signal.png",
            threshold=0.60,
        ),
    ]

    return paths
