#!/usr/bin/env python3
"""Project-side launcher for the installed LeRobot ACT/DP trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


CONFIG_KEYS = {
    "batch_size": "batch_size",
    "learning_rate": "policy.optimizer_lr",
    "chunk_size": "policy.chunk_size",
    "n_action_steps": "policy.n_action_steps",
    "horizon": "policy.horizon",
    "n_observation_steps": "policy.n_obs_steps",
    "evaluation_split": "dataset.eval_split",
}


def _cli_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _config_overrides(path: str | None, policy: str) -> list[str]:
    if path is None:
        return []
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("policy") != policy:
        raise ValueError(f"config policy {config.get('policy')!r} does not match {policy!r}")
    values = []
    for source_key, target_key in CONFIG_KEYS.items():
        if source_key in config:
            values.append(f"--{target_key}={_cli_value(config[source_key])}")
    return values


def _normalize_overrides(overrides: list[str]) -> list[str]:
    normalized = []
    for item in overrides:
        if item.startswith("--optimizer.lr="):
            item = "--policy.optimizer_lr=" + item.split("=", 1)[1]
        normalized.append(item)
    return normalized


def build_command(policy: str, argv: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", "--dataset", dest="dataset_root", required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true")
    # Preserve explicit baseline trainer overrides after "--".
    args, overrides = parser.parse_known_args(
        [item for item in (sys.argv[1:] if argv is None else argv) if item != "--"]
    )
    overrides = _normalize_overrides([item for item in overrides if item != "--"])
    dataset = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"not a LeRobot dataset: {dataset}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    provenance_path = dataset / "meta" / "mimic_conversion.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    repo_id = info.get("repo_id") or provenance.get("lerobot_repo_id")
    if not repo_id:
        raise ValueError("LeRobot meta/info.json has no repo_id")
    config_overrides = _config_overrides(args.config, policy)
    steps = 1 if args.smoke else args.steps
    command = [
        sys.executable, "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={repo_id}", f"--dataset.root={dataset}",
        f"--policy.type={policy}", f"--policy.device={args.device}",
        "--policy.push_to_hub=false", f"--output_dir={output}",
        f"--steps={steps}", f"--seed={args.seed}", "--wandb.enable=false",
        "--env_eval_freq=0", "--eval_steps=0",
    ]
    if args.smoke:
        command.extend(("--batch_size=1", "--num_workers=0", "--log_freq=1", "--save_checkpoint=false"))
        if policy == "act":
            command.extend(("--policy.chunk_size=4", "--policy.n_action_steps=4", "--policy.dim_model=64", "--policy.n_heads=4", "--policy.dim_feedforward=256", "--policy.n_encoder_layers=1", "--policy.pretrained_backbone_weights=null"))
        else:
            command.extend(("--policy.horizon=8", "--policy.n_obs_steps=2", "--policy.n_action_steps=4", "--policy.drop_n_last_frames=0", "--policy.down_dims=[64,128,256]", "--policy.diffusion_step_embed_dim=64", "--policy.num_train_timesteps=10", "--policy.pretrained_backbone_weights=null"))
    # Explicit CLI values follow config values so operators can override a
    # saved experiment snapshot without changing it.
    command.extend(config_overrides)
    command.extend(overrides)
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", "/tmp/mimic-hf-cache")
    lerobot_source = str(Path(os.environ["MIMIC_LEROBOT_ROOT"]) / "src") if os.environ.get("MIMIC_LEROBOT_ROOT") else ""
    project_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = ":".join(filter(None, (lerobot_source, project_root, environment.get("PYTHONPATH", ""))))
    return command, environment


def main_for_policy(policy: str) -> int:
    command, environment = build_command(policy)
    os.execvpe(command[0], command, environment)
    return 0
