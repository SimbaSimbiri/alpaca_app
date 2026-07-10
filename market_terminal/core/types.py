from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


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