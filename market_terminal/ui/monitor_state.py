from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MonitorState:
    latest_log_path: str | None
    latest_order_snapshot_path: str | None
    latest_position_snapshot_path: str | None
    engine_mode: str
    paper_trading_only: str
    symbol: str
    latest_close: str
    probability: str
    threshold: str
    signal: str
    desired_state: str
    action: str
    order_qty: str
    risk_approved: str
    risk_reason: str
    current_position_qty: str
    submitted_order_status: str
    last_lifecycle_stage: str
    last_lifecycle_message: str


def _latest_file(paths: list[Path]) -> Path | None:
    existing_paths = [path for path in paths if path.exists() and path.is_file()]

    if not existing_paths:
        return None

    return max(existing_paths, key=lambda path: path.stat().st_mtime)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


def _read_last_csv_row(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    try:
        with open(path, "r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        if not rows:
            return {}

        return rows[-1]

    except Exception:
        return {}


def _fmt(value: Any, default: str = "-") -> str:
    if value is None:
        return default

    if value == "":
        return default

    return str(value)


def _fmt_float(value: Any, digits: int = 4, default: str = "-") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return default


def load_monitor_state(outputs_dir: str | Path = "outputs") -> MonitorState:
    outputs_path = Path(outputs_dir)

    latest_log_path = _latest_file(
        list(outputs_path.rglob("*paper_trade_log*.json"))
    )

    latest_order_snapshot_path = _latest_file(
        list(outputs_path.rglob("paper_orders_*.csv"))
    )

    latest_position_snapshot_path = _latest_file(
        list(outputs_path.rglob("paper_positions_*.csv"))
    )

    log = _read_json(latest_log_path)
    latest_position = _read_last_csv_row(latest_position_snapshot_path)

    lifecycle_events = log.get("lifecycle_events", [])
    last_lifecycle_event = {}

    if isinstance(lifecycle_events, list) and lifecycle_events:
        possible_event = lifecycle_events[-1]
        if isinstance(possible_event, dict):
            last_lifecycle_event = possible_event

    submitted_order = log.get("submitted_order", {})
    if not isinstance(submitted_order, dict):
        submitted_order = {}

    current_position_qty = log.get("current_position_qty")

    if current_position_qty is None and latest_position:
        current_position_qty = latest_position.get("qty")

    return MonitorState(
        latest_log_path=str(latest_log_path) if latest_log_path else None,
        latest_order_snapshot_path=(
            str(latest_order_snapshot_path) if latest_order_snapshot_path else None
        ),
        latest_position_snapshot_path=(
            str(latest_position_snapshot_path) if latest_position_snapshot_path else None
        ),
        engine_mode=_fmt(log.get("engine_mode"), "one_shot_paper_trade"),
        paper_trading_only=_fmt(log.get("paper_trading_only")),
        symbol=_fmt(log.get("symbol")),
        latest_close=_fmt_float(log.get("latest_close"), digits=2),
        probability=_fmt_float(log.get("probability"), digits=4),
        threshold=_fmt_float(log.get("threshold"), digits=4),
        signal=_fmt(log.get("signal")),
        desired_state=_fmt(log.get("desired_state")),
        action=_fmt(log.get("action")),
        order_qty=_fmt(log.get("order_qty")),
        risk_approved=_fmt(log.get("risk_approved")),
        risk_reason=_fmt(log.get("risk_reason")),
        current_position_qty=_fmt(current_position_qty),
        submitted_order_status=_fmt(submitted_order.get("status")),
        last_lifecycle_stage=_fmt(last_lifecycle_event.get("stage")),
        last_lifecycle_message=_fmt(last_lifecycle_event.get("message")),
    )


def monitor_state_to_lines(state: MonitorState) -> list[str]:
    return [
        "System Monitor",
        "=" * 60,
        f"Engine mode: {state.engine_mode}",
        f"Paper trading only: {state.paper_trading_only}",
        "",
        "Latest Signal",
        "-" * 60,
        f"Symbol: {state.symbol}",
        f"Latest close: {state.latest_close}",
        f"Probability: {state.probability}",
        f"Threshold: {state.threshold}",
        f"Signal: {state.signal}",
        f"Desired state: {state.desired_state}",
        "",
        "Decision and Risk",
        "-" * 60,
        f"Action: {state.action}",
        f"Order quantity: {state.order_qty}",
        f"Current position quantity: {state.current_position_qty}",
        f"Risk approved: {state.risk_approved}",
        f"Risk reason: {state.risk_reason}",
        f"Submitted order status: {state.submitted_order_status}",
        "",
        "Lifecycle",
        "-" * 60,
        f"Latest lifecycle stage: {state.last_lifecycle_stage}",
        f"Latest lifecycle message: {state.last_lifecycle_message}",
        "",
        "Runtime Files",
        "-" * 60,
        f"Latest paper log: {state.latest_log_path or '-'}",
        f"Latest order snapshot: {state.latest_order_snapshot_path or '-'}",
        f"Latest position snapshot: {state.latest_position_snapshot_path or '-'}",
    ]