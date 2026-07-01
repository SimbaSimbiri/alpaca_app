from __future__ import annotations

import numpy as np
import pandas as pd

from market_terminal.backtester import BacktestResult
from market_terminal.constants import INITIAL_CAPITAL, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity / running_max) - 1.0


def trade_win_rate(trades: pd.DataFrame) -> tuple[float, int]:
    """
    Computes win rate from completed round trips.

    a completed trade is a BUY followed by a SELL.
    Open trades at the end of the
    backtest are ignored for win-rate purposes.
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

    volatility = float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_returns) > 1 else float("nan")

    if daily_returns.std(ddof=0) > 0:
        sharpe = float(excess_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = float("nan")

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std(ddof=0)

    if len(downside_returns) > 0 and downside_std > 0:
        sortino = float(excess_returns.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))
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
            formatted[col] = formatted[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2%}")

    for col in ["Sharpe Ratio", "Sortino Ratio"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")

    if "Ending Value" in formatted.columns:
        formatted["Ending Value"] = formatted["Ending Value"].map(lambda x: f"${x:,.2f}")

    return formatted
