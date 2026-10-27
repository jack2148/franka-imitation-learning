"""Non-blocking raw-Mimic to LeRobot conversion lifecycle."""
from __future__ import annotations
import sys

import os
from pathlib import Path
import time

from .common.process import ManagedProcess
from .experiment_paths import inside
from .paths import lerobot_root, shared_runs_root, workspace_root
from runtime.dataset_validation import validate_dataset
from runtime.collection_workspace import research_root


class ConversionAdapter:
    def __init__(self) -> None:
        self.process = ManagedProcess()
        self.output: Path | None = None
        self._validation: dict | None = None

    def start(self, *, input_root: str, output_root: str, task: str = "pick_and_place", dataset_name: str | None = None) -> dict:
        source = inside(input_root, must_exist=True)
        parent = inside(output_root, must_exist=True)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        # The processed dataset is named after its selected raw input root;
        # the timestamp is only for the non-overwriting conversion log.
        name = dataset_name or f"{source.name}_convert"
        if not name or any(not (char.isalnum() or char in "_-") for char in name):
            raise ValueError("dataset_name must contain only letters, numbers, '_' or '-'")
        selected_output = inside(str(parent / name), must_exist=False)
        if selected_output.exists():
            raise FileExistsError(f"processed dataset already exists: {selected_output}")
        raw = validate_dataset(source)
        if raw["type"] not in {"RAW", "RAW_MIMIC"} or not raw["valid"]:
            raise ValueError("conversion input is not a valid Mimic raw dataset root")
        run_id = f"convert_{stamp}"
        python = os.environ.get("MIMIC_PYTHON", sys.executable)
        converter = workspace_root() / "dataset" / "convert_mimic_raw_to_lerobot.py"
        command = (python, str(converter), "--input-root", str(source), "--output-dir", str(selected_output), "--task", task)
        if research_root() is not None and os.environ.get("MIMIC_UI_GEOMAGIC_SOURCE") == "synthetic_udp":
            command += ("--allow-test-data",)
        env = {"PYTHONPATH": f"{lerobot_root() / 'src'}:{workspace_root()}:{os.environ.get('PYTHONPATH', '')}", "HF_HOME": os.environ.get("HF_HOME", "/tmp/mimic-hf-cache")}
        snapshot = self.process.start(command, cwd=workspace_root(), run_id=run_id, output_dir=shared_runs_root() / "conversion" / run_id, environment=env)
        self.output = selected_output
        self._validation = None
        return {"status": snapshot.__dict__, "output": str(selected_output)}

    def state(self) -> dict:
        snapshot = self.process.snapshot()
        if snapshot.state == "COMPLETED" and self.output is not None and self._validation is None:
            self._validation = validate_dataset(self.output)
        return {"status": snapshot.__dict__.copy(), "output": str(self.output) if self.output else None, "validation": self._validation}

    def close(self) -> None:
        self.process.close()
