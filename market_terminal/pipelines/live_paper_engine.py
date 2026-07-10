from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_terminal.core.types import ModelSignal, TradeLifecycleEvent
from market_terminal.execution.alpaca_broker import AlpacaBroker
from market_terminal.pipelines.paper_trade import (
    build_latest_signal,
    build_paper_trade_decision,
    load_model_bundle,
    make_lifecycle_event,
    save_paper_trade_log,
)
from market_terminal.risk.risk_manager import RiskConfig, RiskManager


def run_single_engine_cycle(
    model_bundle_path: Path,
    args: argparse.Namespace,
    broker: AlpacaBroker,
    risk_manager: RiskManager,
) -> Path:
    """
    Runs one model-to-paper-decision cycle for one saved model bundle.

    This function:
    - loads a model bundle
    - downloads fresh market data
    - generates the latest ML signal
    - checks the current paper position
    - builds a BUY/SELL/HOLD decision
    - applies the risk manager
    - optionally submits a paper order
    - writes a structured lifecycle log
    """

    bundle = load_model_bundle(model_bundle_path)

    symbol = str(bundle["symbol"]).upper()
    model = bundle["model"]
    fitted_pca = bundle["fitted_pca"]
    feature_columns = list(bundle["feature_columns"])

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(bundle["probability_threshold"])
    )

    lifecycle_events: list[TradeLifecycleEvent] = []

    raw_data, latest_row, probability, signal = build_latest_signal(
        symbol=symbol,
        model=model,
        fitted_pca=fitted_pca,
        feature_columns=feature_columns,
        threshold=threshold,
        years=args.years,
        feed=args.feed,
        data_delay_minutes=args.data_delay_minutes,
    )

    latest_timestamp = latest_row.index[-1]
    latest_close = float(latest_row["close"].iloc[-1])

    model_signal = ModelSignal(
        symbol=symbol,
        timestamp=latest_timestamp,
        latest_close=latest_close,
        probability=probability,
        threshold=threshold,
        signal=signal,
    )

    lifecycle_events.append(
        make_lifecycle_event(
            stage="signal_generated",
            message="Live paper engine generated latest ML signal.",
            details={
                "symbol": model_signal.symbol,
                "latest_feature_timestamp": str(model_signal.timestamp),
                "latest_close": model_signal.latest_close,
                "probability": model_signal.probability,
                "threshold": model_signal.threshold,
                "signal": model_signal.signal,
                "desired_state": model_signal.desired_state,
            },
        )
    )

    account = broker.get_account()
    buying_power = float(account.buying_power)

    current_qty = broker.get_current_position_qty(symbol)

    decision = build_paper_trade_decision(
        model_signal=model_signal,
        current_position_qty=current_qty,
        requested_qty=float(args.qty),
    )

    lifecycle_events.append(
        make_lifecycle_event(
            stage="decision_built",
            message="Live paper engine built paper-trading decision.",
            details={
                "symbol": decision.symbol,
                "desired_state": decision.desired_state,
                "current_position_qty": decision.current_position_qty,
                "action": decision.action,
                "order_qty": decision.order_qty,
                "reason": decision.reason,
            },
        )
    )

    risk_decision = risk_manager.approve_paper_trade(
        action=decision.action,
        symbol=decision.symbol,
        order_qty=decision.order_qty,
        current_position_qty=decision.current_position_qty,
        latest_price=model_signal.latest_close,
        buying_power=buying_power,
    )

    lifecycle_events.append(
        make_lifecycle_event(
            stage="risk_checked",
            message="Risk manager evaluated live paper engine decision.",
            details={
                "approved": risk_decision.approved,
                "reason": risk_decision.reason,
                "max_order_qty": float(args.max_order_qty),
                "min_buying_power_after_order": float(args.min_buying_power_after_order),
            },
        )
    )

    submitted_order: Any = None

    if decision.action in {"BUY", "SELL"} and args.execute and risk_decision.approved:
        submitted_order = broker.submit_market_order(
            symbol=decision.symbol,
            side=decision.action,
            qty=decision.order_qty,
        )

        lifecycle_events.append(
            make_lifecycle_event(
                stage="order_submitted",
                message="Live paper engine submitted paper order to Alpaca.",
                details={
                    "symbol": decision.symbol,
                    "side": decision.action,
                    "qty": decision.order_qty,
                    "order": AlpacaBroker.serialize_order(submitted_order),
                },
            )
        )

    elif decision.action in {"BUY", "SELL"} and not risk_decision.approved:
        lifecycle_events.append(
            make_lifecycle_event(
                stage="order_rejected",
                message="Live paper engine did not submit order because risk rejected it.",
                details={
                    "action": decision.action,
                    "order_qty": decision.order_qty,
                    "risk_reason": risk_decision.reason,
                },
            )
        )

    elif decision.action in {"BUY", "SELL"} and not args.execute:
        lifecycle_events.append(
            make_lifecycle_event(
                stage="dry_run",
                message="Live paper engine did not submit order because --execute was not provided.",
                details={
                    "action": decision.action,
                    "order_qty": decision.order_qty,
                },
            )
        )

    else:
        lifecycle_events.append(
            make_lifecycle_event(
                stage="no_order_needed",
                message="No order was needed because current paper position already matched model state.",
                details={
                    "action": decision.action,
                    "desired_state": decision.desired_state,
                    "current_position_qty": decision.current_position_qty,
                },
            )
        )

    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_trading_only": True,
        "engine_mode": "live_paper_engine",
        "symbol": symbol,
        "model_bundle": str(model_bundle_path),
        "feed": args.feed,
        "data_delay_minutes": args.data_delay_minutes,
        "latest_feature_timestamp": str(model_signal.timestamp),
        "latest_close": model_signal.latest_close,
        "probability": model_signal.probability,
        "threshold": model_signal.threshold,
        "signal": model_signal.signal,
        "desired_state": model_signal.desired_state,
        "account_status": str(account.status),
        "buying_power": str(account.buying_power),
        "current_position_qty": decision.current_position_qty,
        "action": decision.action,
        "order_qty": decision.order_qty,
        "reason": decision.reason,
        "execute": args.execute,
        "risk_approved": risk_decision.approved,
        "risk_reason": risk_decision.reason,
        "max_order_qty": float(args.max_order_qty),
        "min_buying_power_after_order": float(args.min_buying_power_after_order),
        "lifecycle_events": [event.to_dict() for event in lifecycle_events],
        "submitted_order": AlpacaBroker.serialize_order(submitted_order),
    }

    log_path = save_paper_trade_log(
        log=log,
        model_bundle_path=model_bundle_path,
        symbol=symbol,
    )

    print("\nLive Paper Engine Cycle")
    print("-" * 40)
    print(f"Symbol: {symbol}")
    print(f"Latest close: ${model_signal.latest_close:,.2f}")
    print(f"Probability: {model_signal.probability:.4f}")
    print(f"Signal: {model_signal.signal}")
    print(f"Desired state: {model_signal.desired_state}")
    print(f"Current position qty: {decision.current_position_qty}")
    print(f"Action: {decision.action}")
    print(f"Order qty: {decision.order_qty}")
    print(f"Risk approved: {risk_decision.approved}")
    print(f"Execute: {args.execute}")
    print(f"Log: {log_path}")

    return log_path


def run_live_paper_engine_pipeline(args: argparse.Namespace) -> None:
    """
    Runs the live paper engine for one or more model bundles.

    By default this runs one cycle. Use --cycles N for repeated polling or
    --continuous to run until interrupted with Ctrl+C.
    """

    model_bundle_paths = [Path(path) for path in args.model_bundles]

    broker = AlpacaBroker(paper=True)

    risk_manager = RiskManager(
        RiskConfig(
            max_order_qty=float(args.max_order_qty),
            min_buying_power_after_order=float(args.min_buying_power_after_order),
            allow_short_selling=False,
        )
    )

    print("\nLive Paper Trading Engine")
    print("-" * 40)
    print("This engine uses Alpaca paper trading only.")
    print(f"Model bundles: {len(model_bundle_paths)}")
    print(f"Feed: {args.feed}")
    print(f"Polling interval seconds: {args.polling_interval_seconds}")
    print(f"Cycles: {'continuous' if args.continuous else args.cycles}")
    print(f"Execute orders: {args.execute}")

    cycle_number = 0

    try:
        while True:
            cycle_number += 1

            print("\n" + "=" * 60)
            print(f"Engine cycle {cycle_number}")
            print("=" * 60)

            for model_bundle_path in model_bundle_paths:
                try:
                    run_single_engine_cycle(
                        model_bundle_path=model_bundle_path,
                        args=args,
                        broker=broker,
                        risk_manager=risk_manager,
                    )
                except Exception as exc:
                    print("\nEngine cycle error")
                    print("-" * 40)
                    print(f"Model bundle: {model_bundle_path}")
                    print(f"Error: {exc}")

            if not args.continuous and cycle_number >= args.cycles:
                break

            time.sleep(float(args.polling_interval_seconds))

    except KeyboardInterrupt:
        print("\nLive paper engine stopped by user.")