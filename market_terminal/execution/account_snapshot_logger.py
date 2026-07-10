from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_terminal.execution.alpaca_broker import AlpacaBroker


class AccountSnapshotLogger:
    """
    Saves paper-account order and position snapshots to CSV files.

    These files are runtime artifacts and should stay out of Git.
    """

    ORDER_FIELDS = [
        "snapshot_timestamp",
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
        "canceled_at",
        "expired_at",
        "failed_at",
        "replaced_at",
        "asset_class",
        "order_class",
    ]

    POSITION_FIELDS = [
        "snapshot_timestamp",
        "asset_id",
        "symbol",
        "exchange",
        "asset_class",
        "qty",
        "avg_entry_price",
        "side",
        "market_value",
        "cost_basis",
        "unrealized_pl",
        "unrealized_plpc",
        "unrealized_intraday_pl",
        "unrealized_intraday_plpc",
        "current_price",
        "lastday_price",
        "change_today",
    ]

    def __init__(self, output_dir: str | Path = "outputs/paper_trading") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _date_stamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _write_rows(
        self,
        path: Path,
        fieldnames: list[str],
        rows: list[dict[str, Any]],
    ) -> Path:
        file_exists = path.exists()

        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for row in rows:
                writer.writerow(row)

        return path

    def write_order_snapshot(self, orders: list[Any]) -> Path:
        snapshot_timestamp = self._timestamp()
        path = self.output_dir / f"paper_orders_{self._date_stamp()}.csv"

        rows = []

        for order in orders:
            row = AlpacaBroker.serialize_order(order)
            row["snapshot_timestamp"] = snapshot_timestamp
            rows.append(row)

        if not rows:
            rows.append({"snapshot_timestamp": snapshot_timestamp})

        return self._write_rows(
            path=path,
            fieldnames=self.ORDER_FIELDS,
            rows=rows,
        )

    def write_position_snapshot(self, positions: list[Any]) -> Path:
        snapshot_timestamp = self._timestamp()
        path = self.output_dir / f"paper_positions_{self._date_stamp()}.csv"

        rows = []

        for position in positions:
            row = AlpacaBroker.serialize_position(position)
            row["snapshot_timestamp"] = snapshot_timestamp
            rows.append(row)

        if not rows:
            rows.append({"snapshot_timestamp": snapshot_timestamp})

        return self._write_rows(
            path=path,
            fieldnames=self.POSITION_FIELDS,
            rows=rows,
        )