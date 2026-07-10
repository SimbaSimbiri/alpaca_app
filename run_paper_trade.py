from __future__ import annotations

import argparse

from market_terminal.pipelines.paper_trade import run_paper_trade_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Alpaca paper-trading demo using a saved ML model bundle.",
    )

    parser.add_argument(
        "--model-bundle",
        type=str,
        required=True,
        help="Path to saved model bundle, for example outputs/AAPL_.../artifacts/ml_aapl_model_bundle.joblib.",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Optional symbol override. Defaults to the symbol stored in the model bundle.",
    )

    parser.add_argument(
        "--qty",
        type=float,
        default=1.0,
        help="Share quantity for new BUY paper orders.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional probability threshold override. Defaults to the threshold stored in the model bundle.",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of fresh data to download for latest feature calculation.",
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
        help="Minutes to subtract from current time to avoid recent SIP data restrictions.",
    )

    parser.add_argument(
        "--max-order-qty",
        type=float,
        default=10.0,
        help="Maximum paper order quantity allowed by the risk manager.",
    )
    
    parser.add_argument(
        "--order-snapshot-limit",
        type=int,
        default=50,
        help="Number of recent Alpaca paper orders to include in the order snapshot.",
    )

    parser.add_argument(
        "--min-buying-power-after-order",
        type=float,
        default=0.0,
        help="Minimum buying power required after a paper BUY order.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit the paper order. Without this flag, the script only prints the decision.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_paper_trade_pipeline(args)


if __name__ == "__main__":
    main()