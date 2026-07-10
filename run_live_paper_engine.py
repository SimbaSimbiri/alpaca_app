from __future__ import annotations

import argparse

from market_terminal.pipelines.live_paper_engine import run_live_paper_engine_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runs the live paper-trading engine using saved ML model bundles.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path, for example config/config.yaml.",
    )

    parser.add_argument(
        "--model-bundles",
        nargs="+",
        default=None,
        help=(
            "One or more saved model bundle paths. If omitted, the engine "
            "uses model_bundles from the YAML config."
        ),
    )

    parser.add_argument(
        "--qty",
        type=float,
        default=None,
        help="Share quantity for new BUY paper orders.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional probability threshold override. Defaults to each model bundle threshold.",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Number of years of fresh data to download for latest feature calculation.",
    )

    parser.add_argument(
        "--feed",
        type=str,
        default=None,
        choices=["iex", "sip"],
        help="Alpaca data feed. Defaults to SIP.",
    )

    parser.add_argument(
        "--data-delay-minutes",
        type=int,
        default=None,
        help="Minutes to subtract from current time to avoid recent SIP data restrictions.",
    )

    parser.add_argument(
        "--max-order-qty",
        type=float,
        default=None,
        help="Maximum paper order quantity allowed by the risk manager.",
    )

    parser.add_argument(
        "--order-snapshot-limit",
        type=int,
        default=None,
        help="Number of recent Alpaca paper orders to include in each order snapshot.",
    )

    parser.add_argument(
        "--min-buying-power-after-order",
        type=float,
        default=None,
        help="Minimum buying power required after a paper BUY order.",
    )

    parser.add_argument(
        "--polling-interval-seconds",
        type=float,
        default=None,
        help="Seconds to wait between engine cycles.",
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of engine cycles to run unless --continuous is provided.",
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        default=None,
        help="Run continuously until interrupted with Ctrl+C.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        default=None,
        help="Submit approved paper orders. Without this flag, the engine runs in dry-run mode.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_live_paper_engine_pipeline(args)


if __name__ == "__main__":
    main()