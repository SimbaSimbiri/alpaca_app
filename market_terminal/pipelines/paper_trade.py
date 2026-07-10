from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from market_terminal.core.time_utils import years_ago
from market_terminal.data.alpaca_historical import download_daily_ohlcv_from_alpaca
from market_terminal.execution.account_snapshot_logger import AccountSnapshotLogger
from market_terminal.execution.alpaca_broker import AlpacaBroker
from market_terminal.features.feature_engineering import add_ml_features
from market_terminal.features.pca import transform_pca
from market_terminal.risk.risk_manager import RiskConfig, RiskManager
from market_terminal.strategy.ml_model import predict_up_close_probability, probability_to_signal
from market_terminal.core.types import ModelSignal, PaperTradeDecision, TradeLifecycleEvent

def load_model_bundle(model_bundle_path: Path) -> dict[str, Any]:
    if not model_bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_bundle_path}")

    bundle = joblib.load(model_bundle_path)

    required_keys = {
        "symbol",
        "model",
        "fitted_pca",
        "feature_columns",
        "probability_threshold",
    }

    missing = required_keys.difference(bundle.keys())

    if missing:
        raise ValueError(f"Model bundle is missing required keys: {missing}")

    return bundle


def build_latest_signal(
        symbol: str,
        model,
        fitted_pca,
        feature_columns: list[str],
        threshold: float,
        years: int,
        feed: str,
        data_delay_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float, int]:
    """
    Downloads fresh SIP/IEX data, rebuilds features, applies saved PCA,
    and generates the latest ML signal.

    Uses add_ml_features(), not get_clean_ml_dataset(), because the latest row
    has no known next-day target and should still be usable for live inference.
    """

    now_utc = datetime.now(timezone.utc)
    end = now_utc - timedelta(minutes=data_delay_minutes)
    start = years_ago(end, years)

    df = download_daily_ohlcv_from_alpaca(
        symbol=symbol,
        start=start,
        end=end,
        feed=feed,
    )

    feature_data = add_ml_features(df)
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.dropna(subset=feature_columns).copy()

    if feature_data.empty:
        raise ValueError(
            "No usable feature rows were created. "
            "Try increasing --years so rolling indicators have enough history."
        )

    latest_row = feature_data.tail(1).copy()

    X_latest = latest_row[feature_columns]
    X_latest_pca = transform_pca(X_latest, fitted_pca)

    probability = float(
        predict_up_close_probability(model=model, X=X_latest_pca).iloc[-1]
    )

    signal = int(
        probability_to_signal(
            probabilities=pd.Series(
                [probability],
                index=latest_row.index,
                name="ml_probability",
            ),
            threshold=threshold,
        ).iloc[-1]
    )

    return df, latest_row, probability, signal


def build_paper_trade_decision(
    model_signal: ModelSignal,
    current_position_qty: float,
    requested_qty: float,
) -> PaperTradeDecision:
    """
    Converts a model signal and current paper position into a concrete
    BUY, SELL, or HOLD decision.
    """

    currently_long = current_position_qty > 0

    if model_signal.signal == 1 and not currently_long:
        return PaperTradeDecision(
            symbol=model_signal.symbol,
            desired_state=model_signal.desired_state,
            current_position_qty=current_position_qty,
            action="BUY",
            order_qty=float(requested_qty),
            reason="Model wants LONG and there is currently no paper position.",
        )

    if model_signal.signal == 1 and currently_long:
        return PaperTradeDecision(
            symbol=model_signal.symbol,
            desired_state=model_signal.desired_state,
            current_position_qty=current_position_qty,
            action="HOLD",
            order_qty=0.0,
            reason="Model wants LONG and the paper account is already long.",
        )

    if model_signal.signal == 0 and currently_long:
        return PaperTradeDecision(
            symbol=model_signal.symbol,
            desired_state=model_signal.desired_state,
            current_position_qty=current_position_qty,
            action="SELL",
            order_qty=abs(float(current_position_qty)),
            reason="Model wants FLAT and the paper account currently has a long position.",
        )

    return PaperTradeDecision(
        symbol=model_signal.symbol,
        desired_state=model_signal.desired_state,
        current_position_qty=current_position_qty,
        action="HOLD",
        order_qty=0.0,
        reason="Model wants FLAT and there is currently no paper position.",
    )


def make_lifecycle_event(
    stage: str,
    message: str,
    details: dict | None = None,
) -> TradeLifecycleEvent:
    return TradeLifecycleEvent(
        stage=stage,
        timestamp=datetime.now(timezone.utc),
        message=message,
        details=details or {},
    )

def save_paper_trade_log(
        log: dict[str, Any],
        model_bundle_path: Path,
        symbol: str,
) -> Path:
    """
    Saves the paper-trading decision log beside the model output folder.
    """

    try:
        output_dir = model_bundle_path.parents[1]
        log_dir = output_dir / "paper_trading"
    except IndexError:
        log_dir = Path("outputs") / "paper_trading"

    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"ml_{symbol.lower()}_paper_trade_log_{timestamp}.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(log, file, indent=2, default=str)

    return path


def run_paper_trade_pipeline(args: argparse.Namespace) -> None:
    model_bundle_path = Path(args.model_bundle)

    bundle = load_model_bundle(model_bundle_path)

    symbol = args.symbol.upper() if args.symbol else str(bundle["symbol"]).upper()
    model = bundle["model"]
    fitted_pca = bundle["fitted_pca"]
    feature_columns = list(bundle["feature_columns"])

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(bundle["probability_threshold"])
    )

    print("\nML Paper Trading Demo")
    print("-" * 40)
    print("This is paper trading only — no real money is used.")
    print(f"Symbol: {symbol}")
    print(f"Model bundle: {model_bundle_path}")
    print(f"Data feed: {args.feed}")
    print(f"Data delay minutes: {args.data_delay_minutes}")
    print(f"Probability threshold: {threshold:.2f}")
    print(f"Execute orders: {args.execute}")

    lifecycle_events: list[TradeLifecycleEvent] = []

    # ------------------------------------------------------------
    # 1. Build latest ML signal
    # ------------------------------------------------------------

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
            message="ML model generated latest trading signal.",
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

    print("\nLatest Market/Model State")
    print("-" * 40)
    print(f"Latest feature timestamp: {latest_timestamp}")
    print(f"Latest close: ${latest_close:,.2f}")
    print(f"Predicted probability of positive next-day return: {probability:.4f}")
    print(f"ML signal: {signal}")
    print(f"Desired state: {model_signal.desired_state}")

    # ------------------------------------------------------------
    # 2. Check paper account and current position
    # ------------------------------------------------------------

    broker = AlpacaBroker(paper=True)
    account = broker.get_account()

    current_qty = broker.get_current_position_qty(symbol)

    buying_power = float(account.buying_power)

    risk_manager = RiskManager(
        RiskConfig(
            max_order_qty=float(args.max_order_qty),
            min_buying_power_after_order=float(args.min_buying_power_after_order),
            allow_short_selling=False,
        )
    )

    print("\nPaper Account State")
    print("-" * 40)
    print(f"Account status: {account.status}")
    print(f"Buying power: ${buying_power:,.2f}")
    print(f"Current {symbol} paper position quantity: {current_qty}")

    # ------------------------------------------------------------
    # 3. Decide action
    # ------------------------------------------------------------

    decision = build_paper_trade_decision(
        model_signal=model_signal,
        current_position_qty=current_qty,
        requested_qty=float(args.qty),
    )

    lifecycle_events.append(
        make_lifecycle_event(
            stage="decision_built",
            message="Paper-trading decision built from model signal and current position.",
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

    print("\nTrading Decision")
    print("-" * 40)
    print(f"Action: {decision.action}")
    print(f"Quantity: {decision.order_qty}")
    print(f"Reason: {decision.reason}")

    risk_decision = risk_manager.approve_paper_trade(
        action=decision.action,
        symbol=decision.symbol,
        order_qty=decision.order_qty,
        current_position_qty=decision.current_position_qty,
        latest_price=model_signal.latest_close,
        buying_power=buying_power,
    )

    print("\nRisk Check")
    print("-" * 40)
    print(f"Approved: {risk_decision.approved}")
    print(f"Reason: {risk_decision.reason}")

    lifecycle_events.append(
        make_lifecycle_event(
            stage="risk_checked",
            message="Risk manager evaluated the paper-trading decision.",
            details={
                "approved": risk_decision.approved,
                "reason": risk_decision.reason,
                "max_order_qty": float(args.max_order_qty),
                "min_buying_power_after_order": float(args.min_buying_power_after_order),
            },
        )
    )

    # ------------------------------------------------------------
    # 4. Submit paper order only if --execute is provided
    # ------------------------------------------------------------

    submitted_order = None

    if decision.action in {"BUY", "SELL"} and args.execute and risk_decision.approved:
        submitted_order = broker.submit_market_order(
            symbol=decision.symbol,
            side=decision.action,
            qty=decision.order_qty,
        )

        print("\nSubmitted PAPER order")
        print("-" * 40)
        print(f"Order ID: {submitted_order.id}")
        print(f"Symbol: {submitted_order.symbol}")
        print(f"Side: {submitted_order.side}")
        print(f"Quantity: {submitted_order.qty}")
        print(f"Status: {submitted_order.status}")

        lifecycle_events.append(
            make_lifecycle_event(
                stage="order_submitted",
                message="Paper order was submitted to Alpaca.",
                details={
                    "symbol": decision.symbol,
                    "side": decision.action,
                    "qty": decision.order_qty,
                    "order": AlpacaBroker.serialize_order(submitted_order),
                },
            )
        )

    elif decision.action in {"BUY", "SELL"} and not risk_decision.approved:
        print("\nRisk Rejection")
        print("-" * 40)
        print("No paper order was submitted because the risk manager rejected the decision.")
        print(risk_decision.reason)

        lifecycle_events.append(
            make_lifecycle_event(
                stage="order_rejected",
                message="Paper order was not submitted because risk manager rejected it.",
                details={
                    "action": decision.action,
                    "order_qty": decision.order_qty,
                    "risk_reason": risk_decision.reason,
                },
            )
        )

    elif decision.action in {"BUY", "SELL"} and not args.execute:
        print("\nDry Run")
        print("-" * 40)
        print("No paper order was submitted.")
        print("Run again with --execute to submit this order to Alpaca paper trading.")

        lifecycle_events.append(
            make_lifecycle_event(
                stage="dry_run",
                message="Paper order was not submitted because --execute was not provided.",
                details={
                    "action": decision.action,
                    "order_qty": decision.order_qty,
                },
            )
        )

    else:
        print("\nNo Order Submitted")
        print("-" * 40)
        print("The current paper position already matches the model signal.")

        lifecycle_events.append(
            make_lifecycle_event(
                stage="no_order_needed",
                message="No order was submitted because the current position already matches the model state.",
                details={
                    "action": decision.action,
                    "desired_state": decision.desired_state,
                    "current_position_qty": decision.current_position_qty,
                },
            )
        )

    snapshot_logger = AccountSnapshotLogger()

    order_snapshot_path = None
    position_snapshot_path = None
    snapshot_error = None

    try:
        recent_orders = broker.get_recent_orders(limit=int(args.order_snapshot_limit))
        positions = broker.get_positions()

        order_snapshot_path = snapshot_logger.write_order_snapshot(recent_orders)
        position_snapshot_path = snapshot_logger.write_position_snapshot(positions)

        lifecycle_events.append(
            make_lifecycle_event(
                stage="account_snapshot_saved",
                message="Saved paper account order and position snapshots.",
                details={
                    "order_snapshot_path": str(order_snapshot_path),
                    "position_snapshot_path": str(position_snapshot_path),
                },
            )
        )

    except Exception as exc:
        snapshot_error = str(exc)

        lifecycle_events.append(
            make_lifecycle_event(
                stage="account_snapshot_failed",
                message="Failed to save paper account order or position snapshot.",
                details={
                    "error": snapshot_error,
                },
            )
        )

    # ------------------------------------------------------------
    # 5. Save log
    # ------------------------------------------------------------

    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_trading_only": True,
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
        "order_snapshot_path": str(order_snapshot_path) if order_snapshot_path else None,
        "position_snapshot_path": str(position_snapshot_path) if position_snapshot_path else None,
        "snapshot_error": snapshot_error,
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

    print("\nSaved paper-trading log to:")
    print(log_path)
