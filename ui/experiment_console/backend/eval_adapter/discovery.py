"""Discover trained policies from the experiment workspace layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import workspace_root


POLICY_DIRECTORIES = {
    "act": "act",
    "diffusion_policy": "diffusion_policy",
}

POLICY_CONFIG_TYPES = {
    "act": {"act"},
    "diffusion_policy": {"diffusion", "diffusion_policy"},
}

REQUIRED_PRETRAINED_FILES = ("config.json", "model.safetensors")


def discover_experiments() -> list[str]:
    root = workspace_root() / "experiments"
    if not root.is_dir():
        return []
    return sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir() and (item / "training").is_dir()
    )


def discover_training_runs() -> list[dict[str, Any]]:
    """Return loadable runs using the layout produced by ``TrainAdapter``.

    Policy identity comes from the policy-specific directory and is verified
    against each checkpoint's serialized LeRobot config.  A completion marker
    is intentionally not required: a valid ``pretrained_model`` is loadable.
    """

    result: list[dict[str, Any]] = []
    experiments_root = workspace_root() / "experiments"
    for experiment in discover_experiments():
        experiment_root = experiments_root / experiment
        for policy, directory in POLICY_DIRECTORIES.items():
            policy_root = experiment_root / "training" / directory
            if not policy_root.is_dir():
                continue
            for run_dir in sorted(policy_root.iterdir(), key=lambda path: path.name.lower()):
                if not run_dir.is_dir():
                    continue
                checkpoints = discover_checkpoints(run_dir, policy)
                if not checkpoints:
                    continue
                run_config = _read_json(run_dir / "run_config.json")
                run_status = _read_json(run_dir / "run_status.json")
                result.append(
                    {
                        "experiment": experiment,
                        "policy": policy,
                        "name": run_dir.name,
                        "path": str(run_dir.resolve()),
                        "status": "LOADABLE",
                        "configured_policy": run_config.get("policy"),
                        "dataset": _dataset_provenance(run_config, checkpoints[-1]),
                        "run_status": run_status or None,
                        "checkpoints": checkpoints,
                        "default_checkpoint": checkpoints[-1]["path"],
                    }
                )
    return result


def discover_checkpoints(run_dir: Path, policy: str) -> list[dict[str, Any]]:
    checkpoints_root = run_dir / "artifacts" / "checkpoints"
    if not checkpoints_root.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for step_dir in checkpoints_root.iterdir():
        if not step_dir.is_dir() or not step_dir.name.isdigit():
            continue
        pretrained = step_dir / "pretrained_model"
        config = validate_pretrained_model(pretrained, policy)
        if config is None:
            continue
        result.append(
            {
                "name": step_dir.name,
                "step": int(step_dir.name),
                "path": str(pretrained.resolve()),
                "step_path": str(step_dir.resolve()),
                "policy_type": config["type"],
            }
        )
    return sorted(result, key=lambda item: (item["step"], item["name"]))


def validate_pretrained_model(path: Path, policy: str) -> dict[str, Any] | None:
    if policy not in POLICY_CONFIG_TYPES or not path.is_dir():
        return None
    if any(not (path / name).is_file() for name in REQUIRED_PRETRAINED_FILES):
        return None
    config = _read_json(path / "config.json")
    if config.get("type") not in POLICY_CONFIG_TYPES[policy]:
        return None
    return config


def _dataset_provenance(run_config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    dataset_path = run_config.get("dataset")
    train_config = _read_json(Path(checkpoint["path"]) / "train_config.json")
    dataset = train_config.get("dataset")
    dataset_config = dataset if isinstance(dataset, dict) else {}
    repo_id = dataset_config.get("repo_id")
    source = "run_config.json"
    if not isinstance(dataset_path, str) or not dataset_path:
        if dataset_config:
            dataset_path = dataset_config.get("root")
            source = "pretrained_model/train_config.json"
    if not isinstance(dataset_path, str) or not dataset_path:
        return None
    path = Path(dataset_path).expanduser().resolve()
    info = _read_json(path / "meta" / "info.json")
    return {
        "path": str(path),
        "name": path.name,
        "repo_id": repo_id or info.get("repo_id"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "available": path.is_dir(),
        "source": source,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}
