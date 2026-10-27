"""Safe, non-blocking subprocess lifecycle for console jobs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Sequence


JOB_STATES = {"IDLE", "STARTING", "RUNNING", "STOPPING", "COMPLETED", "FAILED"}
# A processed folder can be up to 128 characters before the policy suffix is
# appended to its fixed training-run name.
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,199}$")


@dataclass(frozen=True)
class ProcessSnapshot:
    state: str
    run_id: str | None = None
    pid: int | None = None
    return_code: int | None = None
    log_path: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


class ManagedProcess:
    """Own one child process and terminate its process group on shutdown."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._snapshot = ProcessSnapshot("IDLE")

    def snapshot(self) -> ProcessSnapshot:
        with self._lock:
            self._poll_locked()
            return self._snapshot

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        run_id: str,
        output_dir: Path,
        environment: dict[str, str] | None = None,
    ) -> ProcessSnapshot:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, numbers, '_' or '-'")
        with self._lock:
            self._poll_locked()
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"job already running: {self._snapshot.run_id}")
            if output_dir.exists():
                raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
            if not cwd.is_dir():
                raise FileNotFoundError(f"job working directory does not exist: {cwd}")
            output_dir.mkdir(parents=True, exist_ok=False)
            log_path = output_dir / "console.log"
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            if environment:
                env.update(environment)
            try:
                self._process = subprocess.Popen(
                    [str(item) for item in command],
                    cwd=str(cwd),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except Exception:
                log.close()
                raise
            finally:
                # The child inherits the file descriptor; close the server copy.
                log.close()
            now = time.time()
            self._snapshot = ProcessSnapshot(
                state="STARTING",
                run_id=run_id,
                pid=self._process.pid,
                log_path=str(log_path),
                started_at=now,
            )
            return self._snapshot

    def stop(self) -> ProcessSnapshot:
        with self._lock:
            self._poll_locked()
            if self._process is None or self._process.poll() is not None:
                return self._snapshot
            self._snapshot = ProcessSnapshot(**{**self._snapshot.__dict__, "state": "STOPPING"})
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            return self._snapshot

    def close(self) -> None:
        with self._lock:
            self._poll_locked()
            if self._process is not None and self._process.poll() is None:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                    self._process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self._poll_locked()

    def _poll_locked(self) -> None:
        if self._process is None:
            return
        code = self._process.poll()
        if code is None:
            if self._snapshot.state == "STARTING":
                self._snapshot = ProcessSnapshot(**{**self._snapshot.__dict__, "state": "RUNNING"})
            return
        state = "COMPLETED" if code == 0 else "FAILED"
        if self._snapshot.state == "STOPPING":
            state = "FAILED"
        self._snapshot = ProcessSnapshot(
            **{**self._snapshot.__dict__, "state": state, "return_code": code, "finished_at": time.time()}
        )
