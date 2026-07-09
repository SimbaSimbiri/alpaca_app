from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from alpaca.data import TimeFrameUnit, TimeFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from market_terminal.core.constants import (
    BEAR_COLOR,
    BULL_COLOR,
    CANDLE_ALPHA,
    CANDLE_WIDTH,
    HISTORICAL_DAYS,
    MIN_CANDLE_BODY_RATIO,
    VOLUME_ALPHA,
    VOLUME_BAND_RATIO,
    VOLUME_GAP_RATIO,
    WICK_LINE_WIDTH, CHART_WINDOW_BARS,
)
from market_terminal.data_connector import AlpacaDataConnector
from market_terminal.live_stream import LiveMarketStream


class MarketTerminalApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Market Data Terminal")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        self.root.minsize(1000, 650)

        self.ui_queue: queue.Queue = queue.Queue()
        self.connector = AlpacaDataConnector()
        self.live_stream: LiveMarketStream | None = None

        self.symbol_var = tk.StringVar(value="MSFT")
        self.status_var = tk.StringVar(value="Ready")

        self.timeframe_amount_var = tk.StringVar(value="15")
        self.timeframe_unit_var = tk.StringVar(value="Minute")

        self.bid_var = tk.StringVar(value="—")
        self.ask_var = tk.StringVar(value="—")
        self.last_var = tk.StringVar(value="—")
        self.quote_time_var = tk.StringVar(value="—")
        self.trade_time_var = tk.StringVar(value="—")

        self.chart_bars: pd.DataFrame | None = None
        self.chart_symbol = ""
        self.chart_timeframe_label = ""
        self.chart_left_index = 0
        self.chart_window_bars = CHART_WINDOW_BARS
        self.chart_range_var = tk.StringVar(value="No historical chart loaded")

        self._create_scrollable_page()
        self._build_layout()
        self._check_authentication()
        self._poll_queue()

    def _build_layout(self) -> None:
        parent = self.page
        top_frame = ttk.Frame(parent, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="Ticker:").pack(side=tk.LEFT)

        symbol_entry = ttk.Entry(
            top_frame,
            textvariable=self.symbol_var,
            width=12,
        )
        symbol_entry.pack(side=tk.LEFT, padx=6)

        ttk.Label(top_frame, text="Timeframe:").pack(side=tk.LEFT, padx=(12, 2))

        timeframe_amount_entry = ttk.Entry(
            top_frame,
            textvariable=self.timeframe_amount_var,
            width=5,
        )
        timeframe_amount_entry.pack(side=tk.LEFT, padx=4)

        timeframe_unit_dropdown = ttk.Combobox(
            top_frame,
            textvariable=self.timeframe_unit_var,
            values=["Minute", "Hour", "Day", "Week", "Month"],
            width=8,
            state="readonly",
        )
        timeframe_unit_dropdown.pack(side=tk.LEFT, padx=4)

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
            parent,
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
            parent,
            text="Historical OHLCV Viewer",
            padding=10,
        )
        chart_frame.pack(side=tk.TOP, fill=tk.X, expand=False, padx=10, pady=8)

        self.figure = Figure(figsize=(13, 6), dpi=100)
        self.price_ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        chart_scroll_frame = ttk.Frame(chart_frame)
        chart_scroll_frame.pack(fill=tk.X, pady=(6, 0))

        self.chart_scrollbar = ttk.Scrollbar(
            chart_scroll_frame,
            orient=tk.HORIZONTAL,
            command=self._on_chart_scroll,
        )
        self.chart_scrollbar.pack(fill=tk.X, expand=True)

        ttk.Label(
            chart_frame,
            textvariable=self.chart_range_var,
        ).pack(anchor="w", pady=(4, 0))

        table_frame = ttk.LabelFrame(
            parent,
            text="Most Recent OHLCV Bars",
            padding=10,
        )
        table_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

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

    def _get_selected_timeframe(self) -> tuple[TimeFrame, str]:
        amount_text = self.timeframe_amount_var.get().strip()
        unit_text = self.timeframe_unit_var.get().strip()

        try:
            amount = int(amount_text)
        except ValueError:
            raise ValueError("Timeframe amount must be a whole number.")

        if amount <= 0:
            raise ValueError("Timeframe amount must be greater than zero.")

        unit_map = {
            "Minute": TimeFrameUnit.Minute,
            "Hour": TimeFrameUnit.Hour,
            "Day": TimeFrameUnit.Day,
            "Week": TimeFrameUnit.Week,
            "Month": TimeFrameUnit.Month
        }

        if unit_text not in unit_map:
            raise ValueError("Invalid timeframe unit selected.")

        time_frame = TimeFrame(amount, unit_map[unit_text])
        label = f"{amount}-{unit_text.lower()}"

        return time_frame, label

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

        try:
            time_frame, timeframe_label = self._get_selected_timeframe()
        except ValueError as exc:
            messagebox.showwarning("Invalid Timeframe", str(exc))
            return

        self.status_var.set(
            f"Loading {HISTORICAL_DAYS} days of {timeframe_label} bars for {symbol}..."
        )

        worker = threading.Thread(
            target=self._historical_worker,
            args=(symbol, time_frame, timeframe_label),
            daemon=True,
        )
        worker.start()

    def _historical_worker(self, symbol: str, time_frame: TimeFrame, timeframe_label: str) -> None:
        try:
            bars = self.connector.get_historical_bars(
                symbol=symbol,
                time_frame=time_frame,
                days=HISTORICAL_DAYS
            )

            self.ui_queue.put(
                {
                    "type": "historical",
                    "symbol": symbol,
                    "bars": bars,
                    "timeframe_label": timeframe_label
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
        timeframe_label = message.get("timeframe_label", "selected timeframe")

        if bars.empty:
            self.status_var.set(f"No historical bars returned for {symbol}")
            return

        self._draw_ohlcv_chart(symbol, bars, timeframe_label)
        self._populate_bars_table(bars)

        self.status_var.set(
            f"Loaded {len(bars):,} {timeframe_label} historical bars for {symbol}"
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

    def _create_scrollable_page(self) -> None:
        self.scroll_canvas = tk.Canvas(
            self.root,
            highlightthickness=0,
        )

        self.scrollbar = ttk.Scrollbar(
            self.root,
            orient="vertical",
            command=self.scroll_canvas.yview,
        )

        self.page = ttk.Frame(self.scroll_canvas)

        self.page_window = self.scroll_canvas.create_window(
            (0, 0),
            window=self.page,
            anchor="nw",
        )

        self.scroll_canvas.configure(
            yscrollcommand=self.scrollbar.set,
        )

        self.scroll_canvas.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        self.scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y,
        )

        self.page.bind(
            "<Configure>",
            lambda event: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            ),
        )

        self.scroll_canvas.bind(
            "<Configure>",
            lambda event: self.scroll_canvas.itemconfig(
                self.page_window,
                width=event.width,
            ),
        )

        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.scroll_canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units",
        )

    def _draw_ohlcv_chart(
            self,
            symbol: str,
            bars: pd.DataFrame,
            timeframe_label: str,
    ) -> None:
        if bars.empty:
            self.figure.clear()
            self.price_ax = self.figure.add_subplot(111)
            self.price_ax.set_title(f"No historical data available for {symbol}")
            self.canvas.draw()
            return

        self.chart_bars = bars.copy()
        self.chart_symbol = symbol
        self.chart_timeframe_label = timeframe_label

        total_bars = len(self.chart_bars)
        self.chart_window_bars = min(CHART_WINDOW_BARS, total_bars)

        self.chart_left_index = max(0, total_bars - self.chart_window_bars)

        self._render_chart_window()

    def _on_chart_scroll(self, *args) -> None:
        if self.chart_bars is None or self.chart_bars.empty:
            return

        total_bars = len(self.chart_bars)
        max_left_index = max(0, total_bars - self.chart_window_bars)

        if max_left_index == 0:
            self.chart_left_index = 0
            self._render_chart_window()
            return

        command = args[0]

        if command == "moveto":
            fraction = float(args[1])
            self.chart_left_index = int(fraction * max_left_index)

        elif command == "scroll":
            amount = int(args[1])
            unit = args[2]

            if unit == "pages":
                step = max(1, int(self.chart_window_bars * 0.8))
            else:
                step = max(1, self.chart_window_bars // 10)

            self.chart_left_index += amount * step

        self.chart_left_index = max(
            0,
            min(self.chart_left_index, max_left_index),
        )

        self._render_chart_window()

    def _render_chart_window(self) -> None:
        if self.chart_bars is None or self.chart_bars.empty:
            return

        bars = self.chart_bars
        symbol = self.chart_symbol
        timeframe_label = self.chart_timeframe_label

        left = self.chart_left_index
        right = min(left + self.chart_window_bars, len(bars))

        plot_df = bars.iloc[left:right].copy()

        self.figure.clear()
        self.price_ax = self.figure.add_subplot(111)

        if plot_df.empty:
            self.price_ax.set_title(f"No historical data available for {symbol}")
            self.canvas.draw()
            return

        # Use local x positions so the visible window is compact.
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

            if raw_body_height < min_body_height:
                candle_low = ((open_price + close_price) / 2) - (
                        min_body_height / 2
                )

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

        scaled_volume = (plot_df["volume"] / max_volume) * volume_band_height

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

        # Day separators and labels for only the visible section.
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
            plot_df.index[pos].strftime("%a %d/%m")
            for pos in day_start_positions
        ]

        self.price_ax.set_xticks(day_start_positions)
        self.price_ax.set_xticklabels(
            day_labels,
            rotation=90,
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

        if price_ticks:
            self.price_ax.set_yticks(price_ticks)

        self.price_ax.text(
            0.01,
            0.085,
            "Volume",
            transform=self.price_ax.transAxes,
            fontsize=8,
            alpha=0.65,
        )

        self.price_ax.set_title(
            f"{symbol} {timeframe_label.title()} OHLCV"
        )
        self.price_ax.set_ylabel("Price")
        self.price_ax.grid(True, alpha=0.18)

        # Update chart scrollbar thumb.
        total_bars = len(bars)
        if total_bars > 0:
            first = left / total_bars
            last = right / total_bars
            self.chart_scrollbar.set(first, last)

        start_label = plot_df.index[0].strftime("%a %d/%m/%y %H:%M")
        end_label = plot_df.index[-1].strftime("%a %d/%m/%y %H:%M")

        self.chart_range_var.set(
            f"Showing {start_label} → {end_label} "
            f"({len(plot_df)} of {len(bars)} bars)"
        )

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
