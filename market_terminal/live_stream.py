from __future__ import annotations

import queue
import threading
from typing import Any

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream

from market_terminal.config import load_settings


FEED_MAP = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
}


class LiveMarketStream:
    """
    Runs Alpaca's live websocket stream in a background thread.
    Sends quote/trade updates back through a queue.
    """

    def __init__(self, symbol: str, output_queue: queue.Queue) -> None:
        settings = load_settings()

        self.symbol = symbol.upper().strip()
        self.output_queue = output_queue
        self.feed = FEED_MAP.get(settings.data_feed, DataFeed.IEX)

        self.stream = StockDataStream(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            feed=self.feed,
        )

        self.thread: threading.Thread | None = None

    async def _quote_handler(self, quote: Any) -> None:
        self.output_queue.put(
            {
                "type": "quote",
                "symbol": quote.symbol,
                "bid": quote.bid_price, # most expensive buyers from orderbook
                "ask": quote.ask_price, # cheapest sellers from orderbook
                "timestamp": quote.timestamp,
            }
        )

    async def _trade_handler(self, trade: Any) -> None:
        self.output_queue.put(
            {
                "type": "trade",
                "symbol": trade.symbol,
                "last": trade.price,
                "timestamp": trade.timestamp,
            }
        )

    def start(self) -> None:
        if not self.symbol:
            raise ValueError("Symbol cannot be empty.")

        self.stream.subscribe_quotes(self._quote_handler, self.symbol)
        self.stream.subscribe_trades(self._trade_handler, self.symbol)

        self.thread = threading.Thread(
            target=self._run_stream,
            daemon=True,
        )

        if self.thread:
            self.thread.name = "LiveMarketStream"
            self.thread.start()

    def _run_stream(self) -> None:
        try:
            self.stream.run()
        except Exception as exc:
            self.output_queue.put(
                {
                    "type": "error",
                    "message": f"Live stream error: {exc}",
                }
            )

    def stop(self) -> None:
        try:
            self.stream.stop()
        except Exception:
            pass