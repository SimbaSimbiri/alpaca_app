from __future__ import annotations

import os
import shlex
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


class EngineProcessController:
    """
    Starts and stops the live paper-trading engine as a local subprocess.

    This is used by the Tkinter system monitor UI. It does not submit orders
    itself. It only launches commands the user provides.
    """

    def __init__(self, log_dir: str | Path = "outputs/ui_engine_control") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._process: subprocess.Popen | None = None
        self._log_handle: TextIO | None = None
        self._log_path: Path | None = None
        self._last_return_code: int | None = None

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    @property
    def last_return_code(self) -> int | None:
        self._cleanup_if_finished()
        return self._last_return_code

    def is_running(self) -> bool:
        self._cleanup_if_finished()
        return self._process is not None and self._process.poll() is None

    def status_text(self) -> str:
        if self.is_running():
            assert self._process is not None
            return f"RUNNING pid={self._process.pid}"

        if self._last_return_code is not None:
            return f"STOPPED return_code={self._last_return_code}"

        return "STOPPED"

    def start(self, command: str) -> Path:
        if self.is_running():
            raise RuntimeError("Engine process is already running.")

        clean_command = command.strip()

        if not clean_command:
            raise ValueError("Engine command cannot be empty.")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_path = self.log_dir / f"engine_process_{timestamp}.log"
        self._log_handle = open(self._log_path, "w", encoding="utf-8")

        self._log_handle.write(f"Command: {clean_command}\n")
        self._log_handle.write(f"Started at: {datetime.now(timezone.utc).isoformat()}\n")
        self._log_handle.write("=" * 80 + "\n")
        self._log_handle.flush()

        args = shlex.split(clean_command)

        creationflags = 0
        preexec_fn = None

        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid

        self._process = subprocess.Popen(
            args,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd(),
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )

        self._last_return_code = None

        return self._log_path

    def stop(self, timeout_seconds: float = 10.0) -> None:
        if self._process is None:
            self._close_log_handle()
            return

        if self._process.poll() is not None:
            self._last_return_code = self._process.returncode
            self._close_log_handle()
            return

        if os.name == "nt":
            self._process.terminate()
        else:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)

        try:
            self._process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=timeout_seconds)

        self._last_return_code = self._process.returncode
        self._close_log_handle()

    def wait(self, timeout: float | None = None) -> int | None:
        if self._process is None:
            return self._last_return_code

        try:
            self._last_return_code = self._process.wait(timeout=timeout)
        finally:
            self._close_log_handle()

        return self._last_return_code

    def _cleanup_if_finished(self) -> None:
        if self._process is None:
            return

        return_code = self._process.poll()

        if return_code is not None:
            self._last_return_code = return_code
            self._close_log_handle()

    def _close_log_handle(self) -> None:
        if self._log_handle is not None and not self._log_handle.closed:
            self._log_handle.write("\n" + "=" * 80 + "\n")
            self._log_handle.write(
                f"Ended at: {datetime.now(timezone.utc).isoformat()}\n"
            )
            self._log_handle.flush()
            self._log_handle.close()