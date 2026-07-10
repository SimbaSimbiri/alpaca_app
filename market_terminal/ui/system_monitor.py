from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from market_terminal.ui.monitor_state import (
    load_monitor_state,
    monitor_state_to_lines,
)


class SystemMonitorApp:
    """
    Lightweight Tkinter UI for monitoring the latest paper-trading state.

    This reads generated runtime logs from outputs/ and does not submit orders.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Alpaca Trading System Monitor")
        self.root.geometry("900x650")

        self.auto_refresh_var = tk.BooleanVar(value=False)
        self.refresh_seconds_var = tk.IntVar(value=10)
        self.status_var = tk.StringVar(value="Ready")

        self._build_layout()
        self.refresh_monitor()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            container,
            text="Paper-Trading System Monitor",
            font=("Segoe UI", 16, "bold"),
        )
        header.pack(anchor=tk.W)

        subtitle = ttk.Label(
            container,
            text=(
                "Displays the latest generated signal, paper-trading decision, "
                "risk check, order status, and account snapshot paths."
            ),
        )
        subtitle.pack(anchor=tk.W, pady=(4, 12))

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(0, 8))

        refresh_button = ttk.Button(
            controls,
            text="Refresh",
            command=self.refresh_monitor,
        )
        refresh_button.pack(side=tk.LEFT)

        auto_refresh_check = ttk.Checkbutton(
            controls,
            text="Auto-refresh",
            variable=self.auto_refresh_var,
            command=self._schedule_auto_refresh,
        )
        auto_refresh_check.pack(side=tk.LEFT, padx=(12, 4))

        ttk.Label(controls, text="Seconds:").pack(side=tk.LEFT, padx=(12, 4))

        refresh_entry = ttk.Entry(
            controls,
            width=6,
            textvariable=self.refresh_seconds_var,
        )
        refresh_entry.pack(side=tk.LEFT)

        self.monitor_text = tk.Text(
            container,
            wrap=tk.WORD,
            height=28,
            font=("Consolas", 10),
        )
        self.monitor_text.pack(fill=tk.BOTH, expand=True)

        status_bar = ttk.Label(
            container,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        status_bar.pack(fill=tk.X, pady=(8, 0))

    def refresh_monitor(self) -> None:
        state = load_monitor_state()
        lines = monitor_state_to_lines(state)

        self.monitor_text.configure(state=tk.NORMAL)
        self.monitor_text.delete("1.0", tk.END)
        self.monitor_text.insert(tk.END, "\n".join(lines))
        self.monitor_text.configure(state=tk.DISABLED)

        self.status_var.set("Monitor refreshed.")

        self._schedule_auto_refresh()

    def _schedule_auto_refresh(self) -> None:
        if not self.auto_refresh_var.get():
            return

        try:
            seconds = max(1, int(self.refresh_seconds_var.get()))
        except Exception:
            seconds = 10
            self.refresh_seconds_var.set(seconds)

        self.root.after(seconds * 1000, self.refresh_monitor)


def main() -> None:
    root = tk.Tk()
    app = SystemMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()