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
    This is to avoid a same-close look-ahead bias where a strategy buys using an
    indicator calculated from a close price that wasn't known until the close.
    """
    if config is None:
        config = BacktestConfig()

    bars = df.copy().sort_index()
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
    Base strategy: buy at the first open and hold to the end.
    """
    if config is None:
        config = BacktestConfig()

    bars = df.copy().sort_index()

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

    trades = pd.DataFrame(trade_records)
    if not trades.empty:
        trades = trades.set_index("timestamp")

    return BacktestResult(name=name, portfolio=portfolio, trades=trades)
