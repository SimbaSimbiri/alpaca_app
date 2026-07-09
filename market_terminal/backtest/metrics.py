from __future__ import annotations

import numpy as np
import pandas as pd

from market_terminal.backtest.engine import BacktestResult
from market_terminal.core.constants import INITIAL_CAPITAL, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity / running_max) - 1.0


def trade_win_rate(trades: pd.DataFrame) -> tuple[float, int]:
    """
    Computes win rate from completed round trips.

    A completed trade is a BUY followed by a SELL.
    Open trades at the end of the backtest are ignored for win-rate purposes.
    """

    if trades.empty:
        return float("nan"), 0

    open_cost: float | None = None
    round_trip_pnls: list[float] = []

    for _, trade in trades.sort_index().iterrows():
        action = str(trade["action"]).upper()
        value = float(trade["price"]) * float(trade["shares"])

        if action == "BUY":
            open_cost = value

        elif action == "SELL" and open_cost is not None:
            round_trip_pnls.append(value - open_cost)
            open_cost = None

    if not round_trip_pnls:
        return float("nan"), 0

    wins = sum(1 for pnl in round_trip_pnls if pnl > 0)

    return wins / len(round_trip_pnls), len(round_trip_pnls)


def calculate_performance_metrics(
    result: BacktestResult,
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float | int | str]:
    portfolio = result.portfolio
    equity = portfolio["equity"].dropna()

    if equity.empty:
        raise ValueError(f"Portfolio equity is empty for {result.name}.")

    daily_returns = equity.pct_change().dropna()
    excess_returns = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)

    ending_value = float(equity.iloc[-1])
    total_return = ending_value / initial_capital - 1.0

    years = max(len(equity) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    cagr = (ending_value / initial_capital) ** (1 / years) - 1.0

    volatility = (
        float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(daily_returns) > 1
        else float("nan")
    )

    if daily_returns.std(ddof=0) > 0:
        sharpe = float(
            excess_returns.mean()
            / daily_returns.std(ddof=0)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = float("nan")

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std(ddof=0)

    if len(downside_returns) > 0 and downside_std > 0:
        sortino = float(
            excess_returns.mean()
            / downside_std
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sortino = float("nan")

    max_drawdown = float(drawdown_series(equity).min())
    win_rate, round_trips = trade_win_rate(result.trades)

    exposure = float((portfolio["shares"] > 0).mean())

    return {
        "Strategy": result.name,
        "Ending Value": ending_value,
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Round Trips": round_trips,
        "Trades": int(len(result.trades)),
        "Exposure": exposure,
    }


def build_metrics_table(
    results: list[BacktestResult],
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    rows = [
        calculate_performance_metrics(
            result=result,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
        )
        for result in results
    ]

    table = pd.DataFrame(rows)

    return table.set_index("Strategy")


def format_metrics_for_console(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()

    percent_columns = [
        "Total Return",
        "CAGR",
        "Volatility",
        "Maximum Drawdown",
        "Win Rate",
        "Exposure",
    ]

    for col in percent_columns:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: "—" if pd.isna(x) else f"{x:.2%}"
            )

    for col in ["Sharpe Ratio", "Sortino Ratio"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: "—" if pd.isna(x) else f"{x:.2f}"
            )

    if "Ending Value" in formatted.columns:
        formatted["Ending Value"] = formatted["Ending Value"].map(
            lambda x: f"${x:,.2f}"
        )

    return formatted


def round_trip_win_rate(round_trips: pd.DataFrame) -> tuple[float, int]:
    """
    Computes win rate from a round-trip DataFrame.

    Expected round_trips columns:
    - pnl_pct

    This works with extract_round_trips_from_result() from backtester.py.
    """

    if round_trips is None or round_trips.empty:
        return float("nan"), 0

    if "pnl_pct" not in round_trips.columns:
        raise ValueError("round_trips must contain a 'pnl_pct' column.")

    wins = (round_trips["pnl_pct"] > 0).sum()
    total = len(round_trips)

    return float(wins / total), int(total)


def calculate_equity_metrics(
    name: str,
    equity: pd.Series,
    daily_returns: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float = RISK_FREE_RATE,
    round_trips: pd.DataFrame | None = None,
    position: pd.Series | None = None,
) -> dict[str, float | int | str]:
    """
    Computes performance metrics directly from an equity curve.

    This is useful for comparison DataFrame where we have:

    - ML portfolio_value
    - ML strategy_returns
    - Buy & Hold value
    - Buy & Hold returns
    """

    equity = equity.dropna().astype(float)
    daily_returns = daily_returns.reindex(equity.index).fillna(0.0).astype(float)

    if equity.empty:
        raise ValueError(f"Equity is empty for {name}.")

    ending_value = float(equity.iloc[-1])
    total_return = ending_value / initial_capital - 1.0

    years = max(len(equity) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    cagr = (ending_value / initial_capital) ** (1 / years) - 1.0

    daily_std = daily_returns.std(ddof=0)

    volatility = (
        float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(daily_returns) > 1
        else float("nan")
    )

    excess_returns = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)

    if daily_std > 0:
        sharpe = float(
            excess_returns.mean()
            / daily_std
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = float("nan")

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std(ddof=0)

    if len(downside_returns) > 0 and downside_std > 0:
        sortino = float(
            excess_returns.mean()
            / downside_std
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sortino = float("nan")

    max_drawdown = float(drawdown_series(equity).min())

    if round_trips is not None:
        win_rate, round_trip_count = round_trip_win_rate(round_trips)
    else:
        win_rate = float("nan")
        round_trip_count = 0

    if position is not None:
        exposure = float(position.reindex(equity.index).fillna(0).mean())
    else:
        exposure = float("nan")

    return {
        "Strategy": name,
        "Ending Value": ending_value,
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Round Trips": round_trip_count,
        "Exposure": exposure,
    }

###
# The following functions are for debugging purposes in the console
###

def build_hw3_performance_table(
    comparison_df: pd.DataFrame,
    round_trips: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Builds the Buy & Hold vs ML Signal performance table.

    Required comparison_df columns:
    - portfolio_value
    - strategy_returns
    - buy_hold_value
    - buy_hold_returns
    - position
    """

    required_columns = {
        "portfolio_value",
        "strategy_returns",
        "buy_hold_value",
        "buy_hold_returns",
        "position",
    }

    missing = required_columns.difference(comparison_df.columns)

    if missing:
        raise ValueError(f"Missing required columns for metrics: {missing}")

    ml_metrics = calculate_equity_metrics(
        name="ML Signal",
        equity=comparison_df["portfolio_value"],
        daily_returns=comparison_df["strategy_returns"],
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        round_trips=round_trips,
        position=comparison_df["position"],
    )

    buy_hold_position = pd.Series(
        1,
        index=comparison_df.index,
        name="buy_hold_position",
    )

    buy_hold_metrics = calculate_equity_metrics(
        name="Buy & Hold",
        equity=comparison_df["buy_hold_value"],
        daily_returns=comparison_df["buy_hold_returns"],
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        round_trips=None,
        position=buy_hold_position,
    )

    table = pd.DataFrame([buy_hold_metrics, ml_metrics])

    return table.set_index("Strategy")


def format_hw3_metrics_for_console(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the metrics table for readable terminal output.
    """

    return format_metrics_for_console(metrics)


def print_hw3_performance_summary(metrics: pd.DataFrame) -> None:
    """
    Prints the performance table and a short interpretation.
    """

    formatted = format_hw3_metrics_for_console(metrics)

    print("\nPerformance Metrics")
    print("-" * 40)
    print(formatted)

    if "Buy & Hold" not in metrics.index or "ML Signal" not in metrics.index:
        return

    ml_return = metrics.loc["ML Signal", "Total Return"]
    buy_hold_return = metrics.loc["Buy & Hold", "Total Return"]

    ml_drawdown = metrics.loc["ML Signal", "Maximum Drawdown"]
    buy_hold_drawdown = metrics.loc["Buy & Hold", "Maximum Drawdown"]

    ml_exposure = metrics.loc["ML Signal", "Exposure"]
    buy_hold_exposure = metrics.loc["Buy & Hold", "Exposure"]

    print("\nInterpretation")
    print("-" * 40)
    print(f"ML Signal total return: {ml_return:.2%}")
    print(f"Buy & Hold total return: {buy_hold_return:.2%}")
    print(f"ML Signal maximum drawdown: {ml_drawdown:.2%}")
    print(f"Buy & Hold maximum drawdown: {buy_hold_drawdown:.2%}")
    print(f"ML Signal exposure: {ml_exposure:.2%}")
    print(f"Buy & Hold exposure: {buy_hold_exposure:.2%}")

    if ml_return > buy_hold_return:
        print("The ML strategy outperformed Buy & Hold on total return.")
    else:
        print("Buy & Hold outperformed the ML strategy on total return.")

    if ml_drawdown > buy_hold_drawdown:
        print("The ML strategy had a smaller maximum drawdown than Buy & Hold.")
    else:
        print("Buy & Hold had a smaller maximum drawdown than the ML strategy.")
