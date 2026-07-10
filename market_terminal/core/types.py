from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from typing import Any

DesiredState = Literal["LONG", "FLAT"]
TradeAction = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class ModelSignal:
    """
    Model-generated signal.

    signal:
    - 1 means the strategy wants to be long
    - 0 means the strategy wants to be flat
    """

    symbol: str
    timestamp: datetime
    latest_close: float
    probability: float
    threshold: float
    signal: int

    @property
    def desired_state(self) -> DesiredState:
        return "LONG" if self.signal == 1 else "FLAT"


@dataclass(frozen=True)
class PaperTradeDecision:
    """
    Decision produced after comparing the model signal with the current paper position.
    """

    symbol: str
    desired_state: DesiredState
    current_position_qty: float
    action: TradeAction
    order_qty: float
    reason: str

@dataclass(frozen=True)
class TradeLifecycleEvent:
    """
    Structured event used for the paper-trading decision lifecycle.

    Example stages:
    - signal_generated
    - decision_built
    - risk_checked
    - dry_run
    - order_submitted
    - order_rejected
    - no_order_needed
    """

    stage: str
    timestamp: datetime
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details or {},
        }