from __future__ import annotations

from dataclasses import dataclass
from math import floor

import pandas as pd

from market_terminal.constants import INITIAL_CAPITAL


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = INITIAL_CAPITAL
    commission_per_trade: float = 0.0
    allow_fractional_shares: bool = False


@dataclass
class BacktestResult:
    name: str
    portfolio: pd.DataFrame
    trades: pd.DataFrame


def _position_size(cash: float, price: float, config: BacktestConfig) -> float:
    available_cash = max(0.0, cash - config.commission_per_trade)

    if price <= 0 or available_cash <= 0:
        return 0.0

    if config.allow_fractional_shares:
        return available_cash / price

    return float(floor(available_cash / price))


def backtest_long_only_strategy(
    df: pd.DataFrame,
    signal: pd.Series,
    name: str,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    The target position produced on day t is executed at the next day's open.
    This avoids same-close look-ahead bias where a strategy buys using an
    indicator calculated from a close price that was not known until the close.

    Rules:
    - signal = 1 means long
    - signal = 0 means flat
    - no short selling
    - no leverage
    - optional commission
    - optional fractional shares
    """
    if config is None:
        config = BacktestConfig()

    bars = df.copy().sort_index()

    required_columns = {"open", "close"}
    missing_columns = required_columns.difference(bars.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns for backtest: {missing_columns}")

    target_signal = signal.reindex(bars.index).fillna(0).astype(int).clip(0, 1)

    cash = float(config.initial_capital)
    shares = 0.0
    trade_records: list[dict] = []
    portfolio_records: list[dict] = []

    for row_number, (timestamp, row) in enumerate(bars.iterrows()):
        open_price = float(row["open"])
        close_price = float(row["close"])

        # Use yesterday's signal for today's execution.
        desired_position = int(target_signal.iloc[row_number - 1]) if row_number > 0 else 0

        action = ""
        trade_price = float("nan")
        trade_shares = 0.0

        if desired_position == 1 and shares == 0:
            qty = _position_size(cash, open_price, config)

            if qty > 0:
                shares = qty
                cash -= qty * open_price + config.commission_per_trade

                action = "BUY"
                trade_price = open_price
                trade_shares = qty

        elif desired_position == 0 and shares > 0:
            qty = shares
            cash += qty * open_price - config.commission_per_trade
            shares = 0.0

            action = "SELL"
            trade_price = open_price
            trade_shares = qty

        holdings = shares * close_price
        equity = cash + holdings

        if action:
            trade_records.append(
                {
                    "timestamp": timestamp,
                    "strategy": name,
                    "action": action,
                    "price": trade_price,
                    "shares": trade_shares,
                    "cash_after": cash,
                    "equity_after": equity,
                }
            )

        portfolio_records.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "shares": shares,
                "holdings": holdings,
                "equity": equity,
                "desired_position": desired_position,
                "raw_signal": int(target_signal.iloc[row_number]),
                "trade_action": action,
            }
        )

    portfolio = pd.DataFrame(portfolio_records).set_index("timestamp")
    portfolio["daily_return"] = portfolio["equity"].pct_change().fillna(0.0)
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1

    trades = pd.DataFrame(trade_records)

    if not trades.empty:
        trades = trades.set_index("timestamp")

    return BacktestResult(name=name, portfolio=portfolio, trades=trades)


def backtest_buy_and_hold(
    df: pd.DataFrame,
    name: str = "Buy & Hold",
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Buy-and-hold benchmark.

    The strategy buys at the first open and holds through the end of the data.
    """

    if config is None:
        config = BacktestConfig()

    bars = df.copy().sort_index()

    required_columns = {"open", "close"}
    missing_columns = required_columns.difference(bars.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns for buy-and-hold backtest: {missing_columns}")

    cash = float(config.initial_capital)
    shares = 0.0
    trade_records: list[dict] = []
    portfolio_records: list[dict] = []

    for row_number, (timestamp, row) in enumerate(bars.iterrows()):
        open_price = float(row["open"])
        close_price = float(row["close"])
        action = ""

        if row_number == 0:
            shares = _position_size(cash, open_price, config)

            if shares > 0:
                cash -= shares * open_price + config.commission_per_trade
                action = "BUY"

                trade_records.append(
                    {
                        "timestamp": timestamp,
                        "strategy": name,
                        "action": action,
                        "price": open_price,
                        "shares": shares,
                        "cash_after": cash,
                        "equity_after": cash + shares * close_price,
                    }
                )

        holdings = shares * close_price
        equity = cash + holdings

        portfolio_records.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "shares": shares,
                "holdings": holdings,
                "equity": equity,
                "desired_position": 1,
                "raw_signal": 1,
                "trade_action": action,
            }
        )

    portfolio = pd.DataFrame(portfolio_records).set_index("timestamp")
    portfolio["daily_return"] = portfolio["equity"].pct_change().fillna(0.0)
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1

    trades = pd.DataFrame(trade_records)

    if not trades.empty:
        trades = trades.set_index("timestamp")

    return BacktestResult(name=name, portfolio=portfolio, trades=trades)

## added long only helpers
def backtest_ml_long_only_signal(
    df: pd.DataFrame,
    signal_col: str = "ml_signal",
    name: str = "ML Signal",
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Expected signal column:
    - 1 = long
    - 0 = flat

    This reuses backtest_long_only_strategy(), so execution still happens at
    the next day's open to avoid look-ahead bias.
    """

    if signal_col not in df.columns:
        raise ValueError(f"Missing ML signal column: {signal_col}")

    signal = df[signal_col].fillna(0).astype(int).clip(0, 1)

    return backtest_long_only_strategy(
        df=df,
        signal=signal,
        name=name,
        config=config,
    )


def extract_round_trips_from_result(
    result: BacktestResult,
    close_open_trade: bool = True,
) -> pd.DataFrame:
    """
    Converts raw BUY/SELL trade logs into completed round trips.

    Raw trades are individual orders.
    Round trips are completed position cycles:

        BUY -> SELL

    The output is useful for win rate and trade-level P&L calculations.
    """

    columns = [
        "strategy",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "shares",
        "pnl_dollars",
        "pnl_pct",
        "holding_days",
    ]

    if result.trades.empty:
        return pd.DataFrame(columns=columns)

    trades = result.trades.copy().sort_index()

    round_trips: list[dict] = []

    entry_date = None
    entry_price = None
    entry_shares = None

    for timestamp, row in trades.iterrows():
        action = row["action"]

        if action == "BUY":
            entry_date = timestamp
            entry_price = float(row["price"])
            entry_shares = float(row["shares"])

        elif action == "SELL" and entry_date is not None:
            exit_date = timestamp
            exit_price = float(row["price"])
            shares = float(row["shares"])

            pnl_dollars = (exit_price - entry_price) * shares
            pnl_pct = exit_price / entry_price - 1

            holding_days = (exit_date - entry_date).days if hasattr(exit_date - entry_date, "days") else None

            round_trips.append(
                {
                    "strategy": result.name,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl_dollars": pnl_dollars,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                }
            )

            entry_date = None
            entry_price = None
            entry_shares = None

    # If still long at the end, close the open position for reporting only.
    # This does not modify the actual backtest portfolio.
    if close_open_trade and entry_date is not None:
        final_date = result.portfolio.index[-1]
        final_price = float(result.portfolio["equity"].iloc[-1])

        final_shares = float(result.portfolio["shares"].iloc[-1])

        if final_shares > 0:
            # Recover final close price from holdings / shares.
            final_close_price = float(result.portfolio["holdings"].iloc[-1] / final_shares)

            pnl_dollars = (final_close_price - entry_price) * entry_shares
            pnl_pct = final_close_price / entry_price - 1

            holding_days = (final_date - entry_date).days if hasattr(final_date - entry_date, "days") else None

            round_trips.append(
                {
                    "strategy": result.name,
                    "entry_date": entry_date,
                    "exit_date": final_date,
                    "entry_price": entry_price,
                    "exit_price": final_close_price,
                    "shares": entry_shares,
                    "pnl_dollars": pnl_dollars,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                }
            )

    return pd.DataFrame(round_trips, columns=columns)


def build_backtest_comparison_frame(
    ml_result: BacktestResult,
    buy_hold_result: BacktestResult,
) -> pd.DataFrame:
    """
    Creates performance metrics and visualizations df.

    Output columns:
    - portfolio_value: ML strategy equity
    - strategy_returns: ML strategy daily returns
    - buy_hold_value: Buy & Hold equity
    - buy_hold_returns: Buy & Hold daily returns
    - drawdown: ML strategy drawdown
    - buy_hold_drawdown: Buy & Hold drawdown
    """

    comparison = pd.DataFrame(index=ml_result.portfolio.index)

    comparison["portfolio_value"] = ml_result.portfolio["equity"]
    comparison["strategy_returns"] = ml_result.portfolio["daily_return"]
    comparison["drawdown"] = ml_result.portfolio["drawdown"]

    buy_hold_aligned = buy_hold_result.portfolio.reindex(comparison.index)

    comparison["buy_hold_value"] = buy_hold_aligned["equity"]
    comparison["buy_hold_returns"] = buy_hold_aligned["daily_return"]
    comparison["buy_hold_drawdown"] = buy_hold_aligned["drawdown"]

    comparison["position"] = (ml_result.portfolio["shares"] > 0).astype(int)
    comparison["trade_action"] = ml_result.portfolio["trade_action"]

    return comparison
