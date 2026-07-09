from __future__ import annotations

import argparse

from market_terminal.pipelines.ml_backtest import run_ml_backtest_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HW3 ML trading pipeline with fresh Alpaca OHLCV data.",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Ticker symbol to backtest, for example AAPL, MSFT, NVDA, SPY.",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date, for example 2021-07-01. Defaults to --years before end.",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date, for example 2026-07-01. Defaults to current UTC time.",
    )

    parser.add_argument(
        "--feed",
        type=str,
        default="sip",
        choices=["iex", "sip"],
        help="Alpaca market data feed. Use 'iex' for free real-time IEX data or 'sip' for delayed/full-market SIP data.",
    )

    parser.add_argument(
        "--data-delay-minutes",
        type=int,
        default=20,
        help="Minutes to subtract from current time to avoid recent SIP data restrictions.",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of data to download when --start is not provided.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Fraction of ML-ready data used for testing.",
    )

    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.80,
        help="Minimum cumulative PCA explained variance threshold.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Long signal threshold. Long if probability > threshold.",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000,
        help="Initial capital for the backtest.",
    )

    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission per trade.",
    )

    parser.add_argument(
        "--allow-fractional-shares",
        action="store_true",
        help="Allow fractional share sizing in the backtest.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root folder where run outputs are saved.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_ml_backtest_pipeline(args)


if __name__ == "__main__":
    main()