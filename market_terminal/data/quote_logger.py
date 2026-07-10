from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarketDataEvent:
    event_type: str
    symbol: str
    event_timestamp: str
    received_timestamp: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    size: float | None = None
    feed: str | None = None


class MarketDataLogger:
    """
    Appends live Alpaca quote/trade events to daily CSV files.

    This is local-file based, so the terminal runs without an external database.
    """

    FIELDNAMES = [
        "received_timestamp",
        "event_timestamp",
        "event_type",
        "symbol",
        "bid",
        "ask",
        "last",
        "size",
        "feed",
    ]

    def __init__(self, output_dir: str | Path = "outputs/live_data") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _daily_path(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.output_dir / f"live_market_events_{today}.csv"

    def log_event(self, event: MarketDataEvent) -> Path:
        path = self._daily_path()
        file_exists = path.exists()

        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "received_timestamp": event.received_timestamp,
                    "event_timestamp": event.event_timestamp,
                    "event_type": event.event_type,
                    "symbol": event.symbol,
                    "bid": event.bid,
                    "ask": event.ask,
                    "last": event.last,
                    "size": event.size,
                    "feed": event.feed,
                }
            )

        return path

    def log_quote(
        self,
        symbol: str,
        bid: float | None,
        ask: float | None,
        event_timestamp: Any,
        feed: str | None = None,
    ) -> Path:
        event = MarketDataEvent(
            event_type="quote",
            symbol=symbol,
            event_timestamp=str(event_timestamp),
            received_timestamp=datetime.now(timezone.utc).isoformat(),
            bid=bid,
            ask=ask,
            feed=feed,
        )

        return self.log_event(event)

    def log_trade(
        self,
        symbol: str,
        last: float | None,
        size: float | None,
        event_timestamp: Any,
        feed: str | None = None,
    ) -> Path:
        event = MarketDataEvent(
            event_type="trade",
            symbol=symbol,
            event_timestamp=str(event_timestamp),
            received_timestamp=datetime.now(timezone.utc).isoformat(),
            last=last,
            size=size,
            feed=feed,
        )

        return self.log_event(event)