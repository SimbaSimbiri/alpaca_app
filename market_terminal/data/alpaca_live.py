from __future__ import annotations

import queue
import threading
from typing import Any
from pathlib import Path

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream

from market_terminal.core.settings import load_settings
from market_terminal.core.constants import DATA_FEED_IEX, DATA_FEED_SIP
from market_terminal.data.quote_logger import MarketDataLogger

FEED_MAP = {
    DATA_FEED_IEX: DataFeed.IEX,
    DATA_FEED_SIP: DataFeed.SIP,
}


class LiveMarketStream:
    """
    Runs Alpaca's live websocket stream in a background thread.
    Sends quote/trade updates back through a queue.
    """

    def __init__(
            self,
            symbol: str,
            output_queue: queue.Queue,
            log_live_data: bool = True,
            log_dir: str | Path = "outputs/live_data",
    ) -> None:
        settings = load_settings()

        self.symbol = symbol.upper().strip()
        self.output_queue = output_queue
        self.feed = FEED_MAP.get(settings.data_feed, DataFeed.IEX)

        self.stream = StockDataStream(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            feed=self.feed,
        )
        self.log_live_data = log_live_data
        self.event_logger = MarketDataLogger(log_dir) if log_live_data else None

        self.thread: threading.Thread | None = None

    async def _quote_handler(self, quote: Any) -> None:
        log_path = None

        if self.event_logger is not None:
            log_path = self.event_logger.log_quote(
                symbol=quote.symbol,
                bid=quote.bid_price,
                ask=quote.ask_price,
                event_timestamp=quote.timestamp,
                feed=str(self.feed),
            )

        self.output_queue.put(
            {
                "type": "quote",
                "symbol": quote.symbol,
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "timestamp": quote.timestamp,
                "log_path": str(log_path) if log_path else None,
            }
        )

    async def _trade_handler(self, trade: Any) -> None:
        trade_size = getattr(trade, "size", None)
        log_path = None

        if self.event_logger is not None:
            log_path = self.event_logger.log_trade(
                symbol=trade.symbol,
                last=trade.price,
                size=trade_size,
                event_timestamp=trade.timestamp,
                feed=str(self.feed),
            )

        self.output_queue.put(
            {
                "type": "trade",
                "symbol": trade.symbol,
                "last": trade.price,
                "size": trade_size,
                "timestamp": trade.timestamp,
                "log_path": str(log_path) if log_path else None,
            }
        )

    def start(self) -> None:
        if not self.symbol:
            raise ValueError("Symbol cannot be empty.")

        # we first subscribe to what we want to stream from before running stream
        # each subscription has a handler that will enable enqueue of events into the app
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