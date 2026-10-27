import sys
"""TRAIN-mode command resolution and lifecycle for existing scripts."""

from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import time

from ..common.process import ManagedProcess
from ..contracts import AdapterStatus, JobRequest
from ..paths import lerobot_root, shared_runs_root, workspace_root
from ..experiment_paths import inside
from runtime.dataset_validation import validate_dataset
from runtime.collection_workspace import research_root


ENTRYPOINTS = {
    "act": "train_act.py",
    "diffusion_policy": "train_diffusion_policy.py",
}


def training_run_name(dataset_root: str | Path, policy: str) -> str:
    """Use the selected converted dataset folder as the training run name."""
    suffix = {"act": "ACT", "diffusion_policy": "Diffusion"}.get(policy)
    if suffix is None:
        raise ValueError(f"unsupported policy: {policy}")
    return f"{Path(dataset_root).name}_{suffix}"


def _entrypoint(policy: str) -> Path:
    return workspace_root() / "scripts" / "training" / ENTRYPOINTS[policy]


class TrainAdapter:
    def __init__(self) -> None:
        self.process = ManagedProcess()

    def status(self) -> AdapterStatus:
        found = [name for name in ENTRYPOINTS if _entrypoint(name).is_file()]
        return AdapterStatus("train", bool(found), "Existing training entry points resolved.", {"policies": found})

    def request(self, policy: str, *, run_id: str | None = None, dataset_root: str | None = None, training_output_root: str | None = None, steps: int | None = None, seed: int = 0, device: str = "cuda", config: str | None = None, overrides: dict[str, object] | None = None) -> JobRequest:
        try:
            entrypoint = ENTRYPOINTS[policy]
        except KeyError as exc:
            raise ValueError(f"unsupported policy: {policy}") from exc
        path = _entrypoint(policy)
        # ManagedProcess creates the run directory for logs first. LeRobot
        # requires its own output_dir not to exist at process start.
        effective_run_id = run_id or (training_run_name(dataset_root, policy) if dataset_root else "preview")
        output_dir = _model_dir(policy, effective_run_id, training_output_root)
        command = [os.environ.get("MIMIC_PYTHON", sys.executable), str(path), "--dataset-root", dataset_root or str(_default_dataset(policy)), "--output-dir", str(output_dir), "--steps", str(steps if steps is not None else 5000), "--seed", str(seed), "--device", device]
        if config:
            command.extend(("--config", config))
        if overrides:
            command.append("--")
            for key, value in overrides.items():
                if not key or key.startswith("-") or any(char.isspace() for char in key):
                    raise ValueError(f"invalid trainer override key: {key!r}")
                command.append(f"--{key}={_cli_value(value)}")
        python_path = f"{lerobot_root() / 'src'}:{workspace_root()}:{os.environ.get('PYTHONPATH', '')}"
        return JobRequest("train", tuple(command), str(workspace_root()), {"CUDA_VISIBLE_DEVICES": "3", "PYTHONPATH": python_path})

    def start(self, policy: str, *, run_id: str | None = None, dataset_root: str | None = None, training_output_root: str | None = None, steps: int | None = None, seed: int = 0, device: str = "cuda", config: str | None = None, overrides: dict[str, object] | None = None) -> dict:
        selected_dataset = str(inside(dataset_root, must_exist=True)) if dataset_root else str(_default_dataset(policy))
        selected_run_id = run_id or training_run_name(selected_dataset, policy)
        validation = validate_dataset(selected_dataset)
        if not validation["training_ready"]:
            raise ValueError(f"dataset is not training-ready: {validation.get('readiness', 'missing LeRobot meta/info.json')}")
        if training_output_root:
            inside(training_output_root, must_exist=True)
        request = self.request(policy, run_id=selected_run_id, dataset_root=selected_dataset, training_output_root=training_output_root, steps=steps, seed=seed, device=device, config=config, overrides=overrides)
        run_dir = _run_dir(policy, selected_run_id, training_output_root)
        if run_dir.exists():
            raise FileExistsError(f"training run already exists: {run_dir}")
        snapshot = self.process.start(
            request.command,
            cwd=Path(request.cwd),
            run_id=selected_run_id,
            output_dir=run_dir,
            environment=request.environment,
        )
        _write_run_config(run_dir, policy=policy, dataset_root=selected_dataset, steps=steps if steps is not None else 5000, seed=seed, device=device, config=config, overrides=overrides or {}, command=request.command)
        return snapshot.__dict__.copy()

    def stop(self) -> dict:
        return self.process.stop().__dict__.copy()

    def state(self) -> dict:
        snapshot = self.process.snapshot().__dict__.copy()
        run_dir = Path(snapshot["log_path"]).parent if snapshot.get("log_path") else None
        return {
            "status": snapshot,
            "log_lines": _read_log_tail(Path(snapshot["log_path"])) if snapshot.get("log_path") else [],
            "dataset_summary": _dataset_summary(run_dir),
            "gpu": _gpu_status(),
            "datasets": self.datasets(),
            "entrypoints": {policy: str(_entrypoint(policy)) for policy in ENTRYPOINTS},
        }

    def datasets(self) -> list[dict]:
        found: dict[str, dict] = {}
        experiments_root = workspace_root() / "experiments"
        roots = list(
            (experiment / "datasets" / "processed")
            for experiment in experiments_root.iterdir()
            if experiment.is_dir()
        ) if experiments_root.is_dir() else []
        if research_root() is not None:
            roots = [research_root() / "processed"]
        for root in roots:
            if not root.is_dir():
                continue
            for info in (root.rglob("meta/info.json") if research_root() else root.glob("*/meta/info.json")):
                try:
                    data = json.loads(info.read_text(encoding="utf-8"))
                    provenance_path = info.parent / "mimic_conversion.json"
                    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
                    stamp = provenance.get("conversion_timestamp")
                    try:
                        converted_at = datetime.fromisoformat(stamp).timestamp() if isinstance(stamp, str) else info.stat().st_mtime
                    except ValueError:
                        converted_at = info.stat().st_mtime
                    path = str(info.parent.parent.resolve())
                    found[path] = {"name": info.parent.parent.name, "path": path, "episodes": data.get("total_episodes"), "frames": data.get("total_frames"), "converted_at": converted_at}
                except (OSError, ValueError):
                    continue
        return sorted(found.values(), key=lambda item: item["converted_at"], reverse=True)

    def close(self) -> None:
        self.process.close()


def _default_dataset(policy: str) -> Path:
    return workspace_root() / "experiments" / "pick_and_place" / "datasets" / "processed"


def _run_dir(policy: str, run_id: str, training_output_root: str | None = None) -> Path:
    parent = inside(training_output_root, must_exist=True) if training_output_root else shared_runs_root() / "training"
    return parent / ("act" if policy == "act" else "diffusion_policy") / run_id


def _model_dir(policy: str, run_id: str, training_output_root: str | None = None) -> Path:
    return _run_dir(policy, run_id, training_output_root) / "artifacts"


def _write_run_config(output_dir: Path, *, policy: str, dataset_root: str, steps: int, seed: int, device: str, config: str | None, overrides: dict[str, object], command: tuple[str, ...]) -> None:
    commit = "unknown"
    try:
        commit = subprocess.run(("git", "-C", str(lerobot_root()), "rev-parse", "HEAD"), capture_output=True, text=True, check=False).stdout.strip() or commit
    except OSError:
        pass
    (output_dir / "run_config.json").write_text(json.dumps({"policy": policy, "dataset": dataset_root, "seed": seed, "steps": steps, "device": device, "config": config, "overrides": overrides, "command": list(command), "git_commit": commit, "timestamp": time.time()}, indent=2) + "\n", encoding="utf-8")


def _cli_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _read_log_tail(path: Path, limit: int = 80) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 32_000))
            text = stream.read().decode("utf-8", errors="replace")
        return text.splitlines()[-limit:]
    except OSError:
        return []


def _dataset_summary(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {"available": False, "path": None, "episodes": None, "frames": None}
    try:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        path = Path(config["dataset"])
        info = json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
        return {"available": True, "path": str(path), "name": path.name, "episodes": info.get("total_episodes"), "frames": info.get("total_frames")}
    except (KeyError, OSError, ValueError):
        return {"available": False, "path": None, "episodes": None, "frames": None}


_GPU_CACHE: tuple[float, dict] = (0.0, {"available": False, "utilization": None, "memory_used": None, "memory_total": None})


def _gpu_status() -> dict:
    global _GPU_CACHE
    now = time.monotonic()
    if now - _GPU_CACHE[0] < 2.0:
        return _GPU_CACHE[1]
    try:
        result = subprocess.run(("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"), capture_output=True, text=True, timeout=0.5, check=False)
        if result.returncode == 0 and result.stdout.strip():
            rows = result.stdout.splitlines()
            if len(rows) <= 3:
                raise ValueError("GPU 3 is not available")
            util, used, total = [int(value.strip()) for value in rows[3].split(",")]
            value = {"available": True, "physical_index": 3, "utilization": util, "memory_used": used, "memory_total": total}
        else:
            value = _GPU_CACHE[1]
    except (OSError, subprocess.SubprocessError, ValueError):
        value = _GPU_CACHE[1]
    _GPU_CACHE = (now, value)
    return value
