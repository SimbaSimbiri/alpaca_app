from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from market_terminal.ui.monitor_state import (
    load_monitor_state,
    monitor_state_to_lines,
)
from market_terminal.ui.engine_process import EngineProcessController


class SystemMonitorApp:
    """
    Lightweight Tkinter UI for monitoring the latest paper-trading state.

    This reads generated runtime logs from outputs/ and does not submit orders.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Trading System Monitor")
        self.root.geometry("900x650")

        self.auto_refresh_var = tk.BooleanVar(value=False)
        self.refresh_seconds_var = tk.IntVar(value=10)
        self.status_var = tk.StringVar(value="Ready")

        self.engine_controller = EngineProcessController()
        self.engine_command_var = tk.StringVar(
            value="python run_live_paper_engine.py --config config/config.yaml --continuous"
        )
        self.engine_status_var = tk.StringVar(value="Engine status: STOPPED")
        self.engine_log_var = tk.StringVar(value="Engine log: -")

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
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

        engine_frame = ttk.LabelFrame(container, text="Live Engine Controls", padding=8)
        engine_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(engine_frame, text="Command:").pack(anchor=tk.W)

        command_entry = ttk.Entry(
            engine_frame,
            textvariable=self.engine_command_var,
        )
        command_entry.pack(fill=tk.X, pady=(2, 8))

        engine_buttons = ttk.Frame(engine_frame)
        engine_buttons.pack(fill=tk.X)

        start_button = ttk.Button(
            engine_buttons,
            text="Start Engine",
            command=self.start_engine,
        )
        start_button.pack(side=tk.LEFT)

        stop_button = ttk.Button(
            engine_buttons,
            text="Stop Engine",
            command=self.stop_engine,
        )
        stop_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            engine_frame,
            textvariable=self.engine_status_var,
        ).pack(anchor=tk.W, pady=(8, 0))

        ttk.Label(
            engine_frame,
            textvariable=self.engine_log_var,
        ).pack(anchor=tk.W)

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

    def start_engine(self) -> None:
        try:
            log_path = self.engine_controller.start(self.engine_command_var.get())
            self.engine_status_var.set(
                f"Engine status: {self.engine_controller.status_text()}"
            )
            self.engine_log_var.set(f"Engine log: {log_path}")
            self.status_var.set("Live paper engine started from UI.")
            self._schedule_engine_status_refresh()
        except Exception as exc:
            self.status_var.set(f"Could not start engine: {exc}")

    def stop_engine(self) -> None:
        try:
            self.engine_controller.stop()
            self.engine_status_var.set(
                f"Engine status: {self.engine_controller.status_text()}"
            )
            self.status_var.set("Live paper engine stopped from UI.")
        except Exception as exc:
            self.status_var.set(f"Could not stop engine: {exc}")

    def _schedule_engine_status_refresh(self) -> None:
        self.engine_status_var.set(
            f"Engine status: {self.engine_controller.status_text()}"
        )

        log_path = self.engine_controller.log_path
        if log_path is not None:
            self.engine_log_var.set(f"Engine log: {log_path}")

        if self.engine_controller.is_running():
            self.root.after(2000, self._schedule_engine_status_refresh)

    def refresh_monitor(self) -> None:
        state = load_monitor_state()
        lines = monitor_state_to_lines(state)

        self.monitor_text.configure(state=tk.NORMAL)
        self.monitor_text.delete("1.0", tk.END)
        self.monitor_text.insert(tk.END, "\n".join(lines))
        self.monitor_text.configure(state=tk.DISABLED)

        self.status_var.set("Monitor refreshed.")
        self.engine_status_var.set(
            f"Engine status: {self.engine_controller.status_text()}"
        )

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

    def on_close(self) -> None:
        if self.engine_controller.is_running():
            self.engine_controller.stop()

        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = SystemMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()