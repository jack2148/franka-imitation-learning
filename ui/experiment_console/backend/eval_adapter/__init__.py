"""EVAL-mode command resolution and lifecycle for closed-loop MuJoCo rollouts."""

from pathlib import Path
import json
import os
import subprocess
import sys

from ..common.process import ManagedProcess
from ..contracts import AdapterStatus, JobRequest
from ..experiment_paths import inside
from ..paths import policy_learning_root, workspace_root
from .discovery import discover_experiments, discover_training_runs


ENTRYPOINTS = {
    "act": "scripts/evaluation/evaluate_policy.py",
    "diffusion_policy": "scripts/evaluation/evaluate_policy.py",
}


def _entrypoint(policy: str) -> Path:
    try:
        relative = ENTRYPOINTS[policy]
    except KeyError as exc:
        raise ValueError(f"unsupported policy: {policy}") from exc
    return policy_learning_root() / relative


class EvalAdapter:
    def __init__(self) -> None:
        self.process = ManagedProcess()

    def status(self) -> AdapterStatus:
        found = [name for name in ENTRYPOINTS if _entrypoint(name).is_file()]
        return AdapterStatus("eval", bool(found), "Closed-loop MuJoCo evaluator resolved.", {"policies": found})

    def request(self, policy: str) -> JobRequest:
        script = _entrypoint(policy)
        if not script.is_file():
            raise FileNotFoundError(f"closed-loop evaluation entrypoint is unavailable: {script}")
        return JobRequest("eval", (sys.executable, str(script)), str(policy_learning_root()))

    def start(
        self,
        policy: str,
        *,
        checkpoint: str,
        experiment: str = "pick_and_place",
        trained_run: str = "",
        dataset_root: str = "",
        run_id: str,
        episodes: int = 1,
        max_steps: int = 2400,
        device: str = "cuda",
        seed: int = 0,
    ) -> dict:
        if not checkpoint:
            raise ValueError("checkpoint is required; no checkpoint default is inferred")
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        script = _entrypoint(policy)
        if not script.is_file():
            raise FileNotFoundError(f"closed-loop evaluation entrypoint is unavailable: {script}")
        selected = _resolve_training_selection(experiment, policy, trained_run, checkpoint)
        resolved_dataset = selected.get("dataset", {}).get("path", "") if selected.get("dataset") else ""
        requested_dataset = str(inside(dataset_root, must_exist=True)) if dataset_root else ""
        if requested_dataset and resolved_dataset and Path(requested_dataset).resolve() != Path(resolved_dataset).resolve():
            raise ValueError("training dataset provenance does not match the selected run")
        output_dir = workspace_root() / "experiments" / experiment / "evaluation" / trained_run / run_id
        command = (
            sys.executable, str(script), "--policy", policy, "--checkpoint", checkpoint,
            "--dataset-provenance", resolved_dataset, "--experiment", experiment, "--trained-run", trained_run,
            "--episodes", str(episodes), "--max-policy-steps", str(max_steps), "--device", device,
            "--seed", str(seed),
            "--output-dir", str(output_dir),
            "--managed-output",
        )
        environment = {"MUJOCO_GL": "egl"}
        # Respect an explicitly selected accelerator, but do not assume a
        # particular physical GPU index on a moved checkout or single-GPU host.
        if "MIMIC_EVAL_CUDA_VISIBLE_DEVICES" in os.environ:
            environment["CUDA_VISIBLE_DEVICES"] = os.environ["MIMIC_EVAL_CUDA_VISIBLE_DEVICES"]
        if "MIMIC_EVAL_EGL_DEVICE_ID" in os.environ:
            environment["MUJOCO_EGL_DEVICE_ID"] = os.environ["MIMIC_EVAL_EGL_DEVICE_ID"]
        snapshot = self.process.start(
            command,
            cwd=workspace_root(),
            run_id=run_id,
            output_dir=output_dir,
            environment=environment,
        )
        _write_eval_config(
            output_dir,
            policy,
            checkpoint,
            resolved_dataset,
            experiment,
            trained_run,
            episodes,
            max_steps,
            device,
            seed,
            command,
        )
        return snapshot.__dict__.copy()

    def action(self, action: str) -> dict:
        if action == "stop":
            return self.stop()
        raise ValueError(f"unsupported eval action: {action}")

    def stop(self) -> dict:
        return self.process.stop().__dict__.copy()

    def state(self) -> dict:
        snapshot = self.process.snapshot().__dict__.copy()
        log_lines = _read_log_tail(Path(snapshot["log_path"])) if snapshot.get("log_path") else []
        run_dir = Path(snapshot["log_path"]).parent if snapshot.get("log_path") else None
        live = _read_json(run_dir / "eval_state.json" if run_dir else None)
        training_runs = discover_training_runs()
        entrypoints = {policy: str(_entrypoint(policy)) for policy in ENTRYPOINTS}
        return {
            "status": snapshot,
            "log_lines": log_lines,
            "live": live,
            "datasets": _datasets(),
            "checkpoints": _checkpoints(training_runs),
            "experiments": discover_experiments(),
            "training_runs": training_runs,
            "result": _result_summary(run_dir),
            "entrypoints": entrypoints,
            "entrypoint_available": {policy: Path(path).is_file() for policy, path in entrypoints.items()},
        }

    def frame_bytes(self) -> bytes | None:
        snapshot = self.process.snapshot().__dict__.copy()
        if not snapshot.get("log_path"):
            return None
        try:
            return (Path(snapshot["log_path"]).parent / "latest_frame.jpg").read_bytes()
        except OSError:
            return None

    def close(self) -> None:
        self.process.close()


def _write_eval_config(
    output_dir: Path,
    policy: str,
    checkpoint: str,
    dataset_root: str,
    experiment: str,
    trained_run: str,
    episodes: int,
    max_steps: int,
    device: str,
    seed: int,
    command: tuple[str, ...],
) -> None:
    try:
        commit = subprocess.run(("git", "-C", str(workspace_root()), "rev-parse", "HEAD"), capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    (output_dir / "evaluation_config.json").write_text(json.dumps({"policy": policy, "checkpoint": checkpoint, "dataset_provenance": dataset_root, "experiment": experiment, "trained_run": trained_run, "episodes": episodes, "max_policy_steps": max_steps, "device": device, "seed": seed, "command": list(command), "git_commit": commit}, indent=2) + "\n", encoding="utf-8")


def _resolve_training_selection(experiment: str, policy: str, trained_run: str, checkpoint: str) -> dict:
    for run in discover_training_runs():
        if run["experiment"] != experiment or run["policy"] != policy or run["name"] != trained_run:
            continue
        checkpoint_path = Path(checkpoint).resolve()
        if not any(Path(item["path"]).resolve() == checkpoint_path for item in run["checkpoints"]):
            raise ValueError("checkpoint is not part of the selected training run")
        return run
    raise ValueError("selected experiment/policy/trained run is not loadable")


def _read_log_tail(path: Path, limit: int = 100) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 40_000))
            return stream.read().decode("utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def _read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _datasets() -> list[dict]:
    result = []
    root = workspace_root() / "experiments"
    infos = (root.glob("*/datasets/processed/*/meta/info.json") if root.is_dir() else ())
    for info in infos:
        try:
            data = json.loads(info.read_text(encoding="utf-8"))
            result.append({"name": info.parent.parent.name, "path": str(info.parent.parent.resolve()), "episodes": data.get("total_episodes"), "frames": data.get("total_frames")})
        except (OSError, ValueError):
            continue
    return sorted(result, key=lambda item: item["name"])


def _checkpoints(training_runs: list[dict] | None = None) -> list[dict]:
    """Flatten the experiment catalog for older API consumers."""
    result = []
    for run in training_runs if training_runs is not None else discover_training_runs():
        for checkpoint in run["checkpoints"]:
            result.append(
                {
                    **checkpoint,
                    "experiment": run["experiment"],
                    "policy": run["policy"],
                    "run": run["name"],
                }
            )
    return result


def _result_summary(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {"available": False, "success_rate": None, "episodes": None}
    for name in ("evaluation_summary.json", "summary.json", "results.json"):
        path = run_dir / name
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {"available": True, "success_rate": data.get("success_rate"), "episodes": data.get("episodes"), "path": str(path)}
            except (OSError, ValueError):
                pass
    return {"available": False, "success_rate": None, "episodes": None}
