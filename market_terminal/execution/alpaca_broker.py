from __future__ import annotations

from typing import Any

from market_terminal.core.settings import get_alpaca_credentials


class AlpacaBroker:
    """
    Thin Alpaca paper-trading broker wrapper.

    Responsibilities:
    - create the Alpaca TradingClient
    - read paper account state
    - read current positions
    - submit paper market orders
    - serialize Alpaca order objects for logs

    Strategy and risk logic should not live here.
    """

    def __init__(self, paper: bool = True) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is not installed. Install it with:\n"
                "pip install alpaca-py"
            ) from exc

        api_key, api_secret = get_alpaca_credentials()

        self.client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper,
        )

    def get_account(self):
        return self.client.get_account()

    def get_current_position_qty(self, symbol: str) -> float:
        """
        Returns current paper position quantity for a symbol.

        If no position exists, returns 0.
        """

        clean_symbol = symbol.upper().strip()

        try:
            position = self.client.get_open_position(clean_symbol)
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
        self,
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

        clean_symbol = symbol.upper().strip()

        if not clean_symbol:
            raise ValueError("Symbol cannot be empty.")

        if qty <= 0:
            raise ValueError("Order quantity must be positive.")

        clean_side = side.upper().strip()

        if clean_side == "BUY":
            order_side = OrderSide.BUY
        elif clean_side == "SELL":
            order_side = OrderSide.SELL
        else:
            raise ValueError("side must be either BUY or SELL.")

        order_request = MarketOrderRequest(
            symbol=clean_symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        return self.client.submit_order(order_data=order_request)

    @staticmethod
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