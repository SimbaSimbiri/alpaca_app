from __future__ import annotations

import argparse
from pathlib import Path

from market_terminal.core.constants import BACKTEST_YEARS, INITIAL_CAPITAL
from market_terminal.pipelines.indicator_backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run technical-indicator strategy backtests using Alpaca historical daily OHLCV data."
    )
    parser.add_argument("--ticker", default="MSFT", help="Ticker symbol, for example AAPL, MSFT, SPY, QQQ, NVDA.")
    parser.add_argument("--years", type=int, default=BACKTEST_YEARS, help="Number of calendar years of daily OHLCV data.")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL, help="Initial capital for each strategy.")
    parser.add_argument("--commission", type=float, default=0.0, help="Flat commission/slippage cost per executed trade.")
    parser.add_argument("--fractional", action="store_true", help="Allow fractional shares. Default uses whole shares.")
    parser.add_argument("--output-root", default="outputs", help="Folder where charts, data, and PDF report are saved.")
    parser.add_argument("--sample", action="store_true", help="Use generated sample data for local testing without Alpaca credentials.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_backtest(
        ticker=args.ticker,
        years=args.years,
        initial_capital=args.initial_capital,
        commission=args.commission,
        fractional=args.fractional,
        output_root=Path(args.output_root),
        use_sample_data=args.sample,
    )


if __name__ == "__main__":
    main()