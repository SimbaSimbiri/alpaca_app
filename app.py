from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from market_terminal.data_connector import AlpacaDataConnector
from market_terminal.live_stream import LiveMarketStream
from market_terminal.constants import (
    BEAR_COLOR,
    BULL_COLOR,
    CANDLE_ALPHA,
    CANDLE_WIDTH,
    DISPLAY_DAYS,
    HISTORICAL_DAYS,
    MIN_CANDLE_BODY_RATIO,
    TIMEFRAME_LABEL,
    VOLUME_ALPHA,
    VOLUME_BAND_RATIO,
    VOLUME_GAP_RATIO,
    WICK_LINE_WIDTH,
)


class MarketTerminalApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Market Data Terminal")
        self.root.geometry("1150x820")

        self.ui_queue: queue.Queue = queue.Queue()
        self.connector = AlpacaDataConnector()
        self.live_stream: LiveMarketStream | None = None

        self.symbol_var = tk.StringVar(value="MSFT")
        self.status_var = tk.StringVar(value="Ready")

        self.bid_var = tk.StringVar(value="—")
        self.ask_var = tk.StringVar(value="—")
        self.last_var = tk.StringVar(value="—")
        self.quote_time_var = tk.StringVar(value="—")
        self.trade_time_var = tk.StringVar(value="—")

        self._build_layout()
        self._check_authentication()
        self._poll_queue()

    def _build_layout(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="Ticker:").pack(side=tk.LEFT)

        symbol_entry = ttk.Entry(
            top_frame,
            textvariable=self.symbol_var,
            width=12,
        )
        symbol_entry.pack(side=tk.LEFT, padx=6)

        ttk.Button(
            top_frame,
            text="Load Historical Data",
            command=self.load_historical_data,
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            top_frame,
            text="Start Live Stream",
            command=self.start_live_stream,
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            top_frame,
            text="Stop Stream",
            command=self.stop_live_stream,
        ).pack(side=tk.LEFT, padx=6)

        ttk.Label(
            top_frame,
            textvariable=self.status_var,
        ).pack(side=tk.LEFT, padx=20)

        quote_frame = ttk.LabelFrame(
            self.root,
            text="Real-Time Market Data",
            padding=10,
        )
        quote_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self._add_quote_box(quote_frame, "Current Bid", self.bid_var, 0)
        self._add_quote_box(quote_frame, "Current Ask", self.ask_var, 1)
        self._add_quote_box(quote_frame, "Last Trade", self.last_var, 2)
        self._add_quote_box(quote_frame, "Quote Time", self.quote_time_var, 3)
        self._add_quote_box(quote_frame, "Trade Time", self.trade_time_var, 4)

        chart_frame = ttk.LabelFrame(
            self.root,
            text="Historical OHLCV Viewer",
            padding=10,
        )
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.figure = Figure(figsize=(11, 5.5), dpi=100)
        self.price_ax = self.figure.add_subplot(211)
        self.volume_ax = self.figure.add_subplot(212, sharex=self.price_ax)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.LabelFrame(
            self.root,
            text="Most Recent OHLCV Bars",
            padding=10,
        )
        table_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        columns = ("timestamp", "open", "high", "low", "close", "volume")
        self.bars_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=8,
        )

        for col in columns:
            self.bars_table.heading(col, text=col.upper())
            self.bars_table.column(col, width=160, anchor=tk.CENTER)

        self.bars_table.pack(fill=tk.X)

    def _add_quote_box(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        column: int,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, padx=16, sticky="w")

        ttk.Label(frame, text=label).pack(anchor="w")

        ttk.Label(
            frame,
            textvariable=variable,
            font=("Arial", 14, "bold"),
        ).pack(anchor="w")

    def _check_authentication(self) -> None:
        try:
            status = self.connector.validate_paper_account()
            self.status_var.set(status)
        except Exception as exc:
            self.status_var.set("Authentication failed")
            messagebox.showerror(
                "Alpaca Authentication Error",
                str(exc),
            )

    def load_historical_data(self) -> None:
        symbol = self.symbol_var.get().upper().strip()

        if not symbol:
            messagebox.showwarning("Missing Symbol", "Please enter a ticker.")
            return

        self.status_var.set(
            f"Loading {HISTORICAL_DAYS} days of {TIMEFRAME_LABEL} bars for {symbol}..."
        )

        worker = threading.Thread(
            target=self._historical_worker,
            args=(symbol,),
            daemon=True,
        )
        worker.start()

    def _historical_worker(self, symbol: str) -> None:
        try:
            bars = self.connector.get_historical_bars(
                symbol=symbol,
                days=HISTORICAL_DAYS,
            )

            self.ui_queue.put(
                {
                    "type": "historical",
                    "symbol": symbol,
                    "bars": bars,
                }
            )

        except Exception as exc:
            self.ui_queue.put(
                {
                    "type": "error",
                    "message": f"Historical data error: {exc}",
                }
            )

    def start_live_stream(self) -> None:
        symbol = self.symbol_var.get().upper().strip()

        if not symbol:
            messagebox.showwarning("Missing Symbol", "Please enter a ticker.")
            return

        self.stop_live_stream()

        self.bid_var.set("—")
        self.ask_var.set("—")
        self.last_var.set("—")
        self.quote_time_var.set("—")
        self.trade_time_var.set("—")

        try:
            self.live_stream = LiveMarketStream(
                symbol=symbol,
                output_queue=self.ui_queue,
            )
            if self.live_stream:
                self.live_stream.start()
                self.status_var.set(
                    f"Stream started for {symbol}. Waiting for live quote/trade events..."
                )
                self.root.after(
                    15000,
                    lambda: self.status_var.set(
                        f"No live events received yet for {symbol}. "
                        "This is common outside market hours."
                    )
                    if self.bid_var.get() == "—" and self.last_var.get() == "—"
                    else None
                )

        except Exception as exc:
            messagebox.showerror("Streaming Error", str(exc))

    def stop_live_stream(self) -> None:
        if self.live_stream:
            self.live_stream.stop()
            self.live_stream = None
            self.status_var.set("Live stream stopped")

    def _poll_queue(self) -> None:
        while not self.ui_queue.empty():
            message = self.ui_queue.get()
            msg_type = message.get("type")

            if msg_type == "historical":
                self._handle_historical_message(message)

            elif msg_type == "quote":
                self._handle_quote_message(message)

            elif msg_type == "trade":
                self._handle_trade_message(message)

            elif msg_type == "error":
                self.status_var.set("Error")
                messagebox.showerror(
                    "Application Error",
                    message.get("message", "Unknown error"),
                )

        self.root.after(500, self._poll_queue)

    def _handle_historical_message(self, message: dict) -> None:
        symbol = message["symbol"]
        bars = message["bars"]

        if bars.empty:
            self.status_var.set(f"No historical bars returned for {symbol}")
            return

        self._draw_ohlcv_chart(symbol, bars)
        self._populate_bars_table(bars)

        self.status_var.set(
            f"Loaded {len(bars):,} {TIMEFRAME_LABEL} historical bars for {symbol}"
        )

    def _handle_quote_message(self, message: dict) -> None:
        bid = message.get("bid")
        ask = message.get("ask")
        timestamp = message.get("timestamp")

        self.bid_var.set(f"${bid:.2f}" if bid is not None else "—")
        self.ask_var.set(f"${ask:.2f}" if ask is not None else "—")
        self.quote_time_var.set(str(timestamp))

        symbol = message.get("symbol", self.symbol_var.get().upper().strip())
        self.status_var.set(f"Received live quote for {symbol}")

    def _handle_trade_message(self, message: dict) -> None:
        last = message.get("last")
        timestamp = message.get("timestamp")

        self.last_var.set(f"${last:.2f}" if last is not None else "—")
        self.trade_time_var.set(str(timestamp))

        symbol = message.get("symbol", self.symbol_var.get().upper().strip())
        self.status_var.set(f"Received live trade for {symbol}")

    def _draw_ohlcv_chart(self, symbol: str, bars: pd.DataFrame) -> None:
        self.figure.clear()
        self.price_ax = self.figure.add_subplot(111)

        latest_timestamp = bars.index.max()
        start_timestamp = latest_timestamp - pd.Timedelta(days=DISPLAY_DAYS)
        plot_df = bars.loc[bars.index >= start_timestamp].copy()

        if plot_df.empty:
            self.price_ax.set_title(f"No historical data available for {symbol}")
            self.canvas.draw()
            return

        x_positions = list(range(len(plot_df)))

        price_min = plot_df["low"].min()
        price_max = plot_df["high"].max()
        price_range = price_max - price_min

        if price_range == 0:
            price_range = max(price_max * 0.01, 1)

        volume_band_height = price_range * VOLUME_BAND_RATIO
        volume_gap = price_range * VOLUME_GAP_RATIO
        volume_base = price_min - volume_band_height - volume_gap

        max_volume = plot_df["volume"].max()
        if max_volume == 0:
            max_volume = 1

        min_body_height = price_range * MIN_CANDLE_BODY_RATIO

        candle_colors = []

        for x_pos, (_, row) in zip(x_positions, plot_df.iterrows()):
            open_price = row["open"]
            high_price = row["high"]
            low_price = row["low"]
            close_price = row["close"]

            is_bullish = close_price >= open_price
            color = BULL_COLOR if is_bullish else BEAR_COLOR
            candle_colors.append(color)

            # Wick
            self.price_ax.plot(
                [x_pos, x_pos],
                [low_price, high_price],
                color=color,
                linewidth=WICK_LINE_WIDTH,
                zorder=3,
            )

            # Candle body
            candle_low = min(open_price, close_price)
            raw_body_height = abs(close_price - open_price)

            candle_height = max(raw_body_height, min_body_height)

            # Center the artificial minimum-height body around the true close/open area.
            if raw_body_height < min_body_height:
                candle_low = ((open_price + close_price) / 2) - (min_body_height / 2)

            rectangle = Rectangle(
                (x_pos - CANDLE_WIDTH / 2, candle_low),
                CANDLE_WIDTH,
                candle_height,
                facecolor=color,
                edgecolor=color,
                alpha=CANDLE_ALPHA,
                linewidth=0.7,
                zorder=4,
            )

            self.price_ax.add_patch(rectangle)

        scaled_volume = (
                                plot_df["volume"] / max_volume
                        ) * volume_band_height

        self.price_ax.bar(
            x_positions,
            scaled_volume,
            bottom=volume_base,
            width=CANDLE_WIDTH,
            color=candle_colors,
            alpha=VOLUME_ALPHA,
            zorder=1,
        )

        self.price_ax.axhline(
            price_min,
            color="gray",
            linewidth=0.8,
            alpha=0.35,
            zorder=2,
        )

        # Day separators and labels.
        normalized_dates = plot_df.index.normalize()
        day_start_positions = []

        for i in range(len(plot_df)):
            if i == 0 or normalized_dates[i] != normalized_dates[i - 1]:
                day_start_positions.append(i)

        for pos in day_start_positions:
            self.price_ax.axvline(
                pos - 0.5,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.30,
                zorder=0,
            )

        day_labels = [
            plot_df.index[pos].strftime("%a %d/%m/%y")
            for pos in day_start_positions
        ]

        self.price_ax.set_xticks(day_start_positions)
        self.price_ax.set_xticklabels(
            day_labels,
            rotation=30,
            ha="right",
            fontsize=8,
        )

        self.price_ax.set_xlim(-1, len(plot_df))
        self.price_ax.set_ylim(
            volume_base,
            price_max + price_range * 0.06,
        )

        # Keep y-axis focused on price values, not the scaled volume region.
        current_ticks = self.price_ax.get_yticks()
        price_ticks = [
            tick for tick in current_ticks
            if price_min <= tick <= price_max
        ]
        self.price_ax.set_yticks(price_ticks)

        self.price_ax.text(
            0.01,
            0.035,
            "Volume scaled visually",
            transform=self.price_ax.transAxes,
            fontsize=8,
            alpha=0.65,
        )

        self.price_ax.set_title(
            f"{symbol} {TIMEFRAME_LABEL.title()} OHLCV — Last {DISPLAY_DAYS} Calendar Days"
        )
        self.price_ax.set_ylabel("Price")

        self.price_ax.grid(True, alpha=0.18)

        self.figure.tight_layout()
        self.canvas.draw()

    def _populate_bars_table(self, bars: pd.DataFrame) -> None:
        for row_id in self.bars_table.get_children():
            self.bars_table.delete(row_id)

        recent_bars = bars.tail(10).sort_index(ascending=False)

        for timestamp, row in recent_bars.iterrows():
            self.bars_table.insert(
                "",
                tk.END,
                values=(
                    str(timestamp),
                    f"{row['open']:.2f}",
                    f"{row['high']:.2f}",
                    f"{row['low']:.2f}",
                    f"{row['close']:.2f}",
                    int(row["volume"]),
                ),
            )

    def on_close(self) -> None:
        self.stop_live_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = MarketTerminalApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()