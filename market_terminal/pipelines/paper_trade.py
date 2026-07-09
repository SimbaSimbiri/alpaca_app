from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from market_terminal.core.settings import get_alpaca_credentials
from market_terminal.core.time_utils import years_ago
from market_terminal.data.alpaca_historical import download_daily_ohlcv_from_alpaca
from market_terminal.features.feature_engineering import add_ml_features
from market_terminal.features.pca import transform_pca
from market_terminal.strategy.ml_model import predict_up_probability, probability_to_signal


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


def get_trading_client():
    """
    Creates an Alpaca paper trading client.

    This is paper trading only. No real money is used.
    """

    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is not installed. Install it with:\n"
            "pip install alpaca-py"
        ) from exc

    api_key, api_secret = get_alpaca_credentials()

    return TradingClient(
        api_key=api_key,
        secret_key=api_secret,
        paper=True,
    )


def get_current_position_qty(trading_client, symbol: str) -> float:
    """
    Returns current paper position quantity for a symbol.

    If no position exists, returns 0.
    """

    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)

    except Exception as exc:
        message = str(exc).lower()

        no_position_messages = [
            "position does not exist",
            "404",
            "not found",
        ]

        if any(text in message for text in no_position_messages):
            return 0.0

        raise


def submit_market_order(
    trading_client,
    symbol: str,
    side: str,
    qty: float,
):
    """
    Submits a paper market order.
    """

    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is not installed. Install it with:\n"
            "pip install alpaca-py"
        ) from exc

    if qty <= 0:
        raise ValueError("Order quantity must be positive.")

    if side.upper() == "BUY":
        order_side = OrderSide.BUY
    elif side.upper() == "SELL":
        order_side = OrderSide.SELL
    else:
        raise ValueError("side must be either BUY or SELL.")

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )

    return trading_client.submit_order(order_data=order_request)


def serialize_order(order) -> dict[str, Any]:
    """
    Converts an Alpaca order object into a JSON-friendly dictionary.
    """

    if order is None:
        return {}

    fields = [
        "id",
        "client_order_id",
        "symbol",
        "side",
        "qty",
        "filled_qty",
        "type",
        "time_in_force",
        "status",
        "submitted_at",
        "filled_at",
    ]

    output = {}

    for field in fields:
        value = getattr(order, field, None)

        if value is not None:
            output[field] = str(value)

    return output


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
        predict_up_probability(
            model=model,
            X=X_latest_pca,
        ).iloc[-1]
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
    path = log_dir / f"hw3_{symbol.lower()}_paper_trade_log_{timestamp}.json"

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

    print("\nHW3 Paper Trading Demo")
    print("-" * 40)
    print("This is paper trading only — no real money is used.")
    print(f"Symbol: {symbol}")
    print(f"Model bundle: {model_bundle_path}")
    print(f"Data feed: {args.feed}")
    print(f"Data delay minutes: {args.data_delay_minutes}")
    print(f"Probability threshold: {threshold:.2f}")
    print(f"Execute orders: {args.execute}")

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

    desired_state = "LONG" if signal == 1 else "FLAT"

    print("\nLatest Market/Model State")
    print("-" * 40)
    print(f"Latest feature timestamp: {latest_timestamp}")
    print(f"Latest close: ${latest_close:,.2f}")
    print(f"Predicted probability of positive next-day return: {probability:.4f}")
    print(f"ML signal: {signal}")
    print(f"Desired state: {desired_state}")

    # ------------------------------------------------------------
    # 2. Check paper account and current position
    # ------------------------------------------------------------

    trading_client = get_trading_client()
    account = trading_client.get_account()

    current_qty = get_current_position_qty(
        trading_client=trading_client,
        symbol=symbol,
    )

    currently_long = current_qty > 0

    print("\nPaper Account State")
    print("-" * 40)
    print(f"Account status: {account.status}")
    print(f"Buying power: ${float(account.buying_power):,.2f}")
    print(f"Current {symbol} paper position quantity: {current_qty}")

    # ------------------------------------------------------------
    # 3. Decide action
    # ------------------------------------------------------------

    action = "HOLD"
    order_qty = 0.0
    reason = ""

    if signal == 1 and not currently_long:
        action = "BUY"
        order_qty = float(args.qty)
        reason = "Model wants LONG and there is currently no paper position."

    elif signal == 1 and currently_long:
        action = "HOLD"
        reason = "Model wants LONG and the paper account is already long."

    elif signal == 0 and currently_long:
        action = "SELL"
        order_qty = abs(float(current_qty))
        reason = "Model wants FLAT and the paper account currently has a long position."

    elif signal == 0 and not currently_long:
        action = "HOLD"
        reason = "Model wants FLAT and there is currently no paper position."

    print("\nTrading Decision")
    print("-" * 40)
    print(f"Action: {action}")
    print(f"Quantity: {order_qty}")
    print(f"Reason: {reason}")

    # ------------------------------------------------------------
    # 4. Submit paper order only if --execute is provided
    # ------------------------------------------------------------

    submitted_order = None

    if action in {"BUY", "SELL"} and args.execute:
        submitted_order = submit_market_order(
            trading_client=trading_client,
            symbol=symbol,
            side=action,
            qty=order_qty,
        )

        print("\nSubmitted PAPER order")
        print("-" * 40)
        print(f"Order ID: {submitted_order.id}")
        print(f"Symbol: {submitted_order.symbol}")
        print(f"Side: {submitted_order.side}")
        print(f"Quantity: {submitted_order.qty}")
        print(f"Status: {submitted_order.status}")

    elif action in {"BUY", "SELL"} and not args.execute:
        print("\nDry Run")
        print("-" * 40)
        print("No paper order was submitted.")
        print("Run again with --execute to submit this order to Alpaca paper trading.")

    else:
        print("\nNo Order Submitted")
        print("-" * 40)
        print("The current paper position already matches the model signal.")

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
        "latest_feature_timestamp": str(latest_timestamp),
        "latest_close": latest_close,
        "probability": probability,
        "threshold": threshold,
        "signal": signal,
        "desired_state": desired_state,
        "account_status": str(account.status),
        "buying_power": str(account.buying_power),
        "current_position_qty": current_qty,
        "action": action,
        "order_qty": order_qty,
        "reason": reason,
        "execute": args.execute,
        "submitted_order": serialize_order(submitted_order),
    }

    log_path = save_paper_trade_log(
        log=log,
        model_bundle_path=model_bundle_path,
        symbol=symbol,
    )

    print("\nSaved paper-trading log to:")
    print(log_path)
