"""Ubuntu-native folder selection constrained to the experiment workspace."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any

from .experiment_paths import EXPERIMENTS_ROOT, folder_listing, inside
from .task_profiles import purpose_defaults


PURPOSE_DEFAULTS = purpose_defaults()

PURPOSE_TITLES = {
    "dataset_output": "Select Dataset Output Folder",
    "raw_input": "Select Raw Dataset Folder",
    "processed_output": "Select Processed Dataset Output Folder",
    "training_input": "Select Training Dataset Folder",
    "training_output": "Select Training Output Folder",
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def parse_zenity_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Convert Zenity exit status into a selection/cancel result."""
    if result.returncode == 1:
        return {"cancelled": True}
    if result.returncode != 0:
        detail = result.stderr.strip() or f"zenity exited with status {result.returncode}"
        raise RuntimeError(f"native folder chooser failed: {detail}")
    selected = result.stdout.strip()
    if not selected:
        raise RuntimeError("native folder chooser returned an empty selection")
    return {"cancelled": False, "path": selected}


class NativeFolderPicker:
    """Launch one Zenity directory chooser at a time in the backend session."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        runner: Runner = subprocess.run,
    ) -> None:
        self.executable = executable or shutil.which("zenity")
        self._runner = runner
        self._guard = threading.Lock()
        self._process_lock = threading.Lock()
        self._process = None
        self._closed = False

    def close(self) -> None:
        with self._process_lock:
            self._closed = True
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=1)

    def _run_dialog(self, command, **kwargs):
        if self._runner is not subprocess.run:
            return self._runner(command, **kwargs)
        with self._process_lock:
            if self._closed:return subprocess.CompletedProcess(command,1,'','')
            self._process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                           text=True,env=kwargs.get('env'))
            process=self._process
        try:
            stdout,stderr=process.communicate()
            if self._closed:return subprocess.CompletedProcess(command,1,'','')
            return subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
        finally:
            with self._process_lock:self._process=None

    def select(self, purpose: str, initial_path: str | None = None) -> dict[str, Any]:
        defaults = purpose_defaults()
        if purpose not in defaults:
            raise ValueError(f"unsupported folder picker purpose: {purpose}")
        if not self.executable:
            raise RuntimeError("zenity is not installed; native folder chooser is unavailable")
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("backend has no accessible Ubuntu GUI display session")
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("a native folder chooser is already open")
        try:
            start = self._initial_directory(purpose, initial_path, defaults)
            result = self._run_dialog(
                (
                    self.executable,
                    "--file-selection",
                    "--directory",
                    f"--title={PURPOSE_TITLES[purpose]}",
                    f"--filename={str(start).rstrip('/')}/",
                ),
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
            parsed = parse_zenity_result(result)
            if parsed["cancelled"]:
                return parsed
            selected = inside(parsed["path"], must_exist=True)
            details = folder_listing(str(selected))
            return {
                "cancelled": False,
                "path": str(selected),
                "writable": details["writable"],
                "free_bytes": details["free_bytes"],
            }
        finally:
            self._guard.release()

    @staticmethod
    def _initial_directory(
        purpose: str, initial_path: str | None, defaults: dict[str, str] | None = None
    ) -> Path:
        if initial_path:
            try:
                return inside(initial_path, must_exist=True)
            except ValueError:
                pass
        default = EXPERIMENTS_ROOT / (defaults or purpose_defaults())[purpose]
        try:
            return inside(str(default), must_exist=True)
        except ValueError:
            return EXPERIMENTS_ROOT
