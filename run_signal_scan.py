from __future__ import annotations

import argparse

from market_terminal.pipelines.signal_scan import (
    DEFAULT_SYMBOLS,
    run_signal_scan_pipeline,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan multiple tickers for current ML long signals.",
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Ticker symbols to scan.",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of fresh data to use.",
    )

    parser.add_argument(
        "--feed",
        type=str,
        default="sip",
        choices=["iex", "sip"],
        help="Alpaca data feed. Defaults to SIP.",
    )

    parser.add_argument(
        "--data-delay-minutes",
        type=int,
        default=20,
        help="Minutes to subtract from current time to avoid recent SIP restrictions.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Long signal threshold.",
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
        help="PCA explained variance threshold.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root output folder.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_signal_scan_pipeline(args)


if __name__ == "__main__":
    main()
