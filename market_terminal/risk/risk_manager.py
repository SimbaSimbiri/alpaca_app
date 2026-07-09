from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


@dataclass(frozen=True)
class RiskConfig:
    max_order_qty: float = 10.0
    min_buying_power_after_order: float = 0.0
    allow_short_selling: bool = False


class RiskManager:
    """
    Paper-trading risk gate.

    Responsibilities:
    - block invalid order quantities
    - block oversized orders
    - block short-selling behavior
    - block buys that exceed available buying power
    - approve HOLD decisions without submitting orders

    This class should not generate signals and should not submit orders.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def approve_paper_trade(
        self,
        action: str,
        symbol: str,
        order_qty: float,
        current_position_qty: float,
        latest_price: float,
        buying_power: float,
    ) -> RiskDecision:
        clean_action = action.upper().strip()
        clean_symbol = symbol.upper().strip()

        if not clean_symbol:
            return RiskDecision(
                approved=False,
                reason="Risk rejected trade because symbol is empty.",
            )

        if clean_action == "HOLD":
            return RiskDecision(
                approved=True,
                reason="Risk approved HOLD because no order will be submitted.",
            )

        if clean_action not in {"BUY", "SELL"}:
            return RiskDecision(
                approved=False,
                reason=f"Risk rejected unsupported action: {action}.",
            )

        if order_qty <= 0:
            return RiskDecision(
                approved=False,
                reason="Risk rejected order because quantity must be positive.",
            )

        if order_qty > self.config.max_order_qty:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Risk rejected order because quantity {order_qty} exceeds "
                    f"max_order_qty {self.config.max_order_qty}."
                ),
            )

        if latest_price <= 0:
            return RiskDecision(
                approved=False,
                reason="Risk rejected order because latest price must be positive.",
            )

        estimated_order_value = order_qty * latest_price

        if clean_action == "BUY":
            estimated_remaining_buying_power = buying_power - estimated_order_value

            if estimated_remaining_buying_power < self.config.min_buying_power_after_order:
                return RiskDecision(
                    approved=False,
                    reason=(
                        "Risk rejected BUY because estimated remaining buying power "
                        f"would be ${estimated_remaining_buying_power:,.2f}, below "
                        f"required minimum ${self.config.min_buying_power_after_order:,.2f}."
                    ),
                )

        if clean_action == "SELL":
            if not self.config.allow_short_selling and order_qty > current_position_qty:
                return RiskDecision(
                    approved=False,
                    reason=(
                        "Risk rejected SELL because order quantity exceeds current "
                        "long position and short selling is disabled."
                    ),
                )

        return RiskDecision(
            approved=True,
            reason="Risk approved paper-trading order.",
        )