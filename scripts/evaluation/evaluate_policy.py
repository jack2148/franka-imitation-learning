#!/usr/bin/env python3
"""Run real LeRobot policy inference against the current FR3 MuJoCo task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_ROOT = PROJECT_ROOT / "environments" / "mimic_robot"
for path in (PROJECT_ROOT, MIMIC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runtime.policy_evaluation.environment import PickPlaceEvaluationEnvironment
from runtime.policy_evaluation.policy import LeRobotPolicyAdapter
from runtime.policy_evaluation.runner import EvaluationConfig, EvaluationRunner, claim_output_directory


STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("act", "diffusion_policy"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-provenance", default="")
    parser.add_argument("--experiment", default="pick_and_place")
    parser.add_argument("--trained-run", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-policy-steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--managed-output", action="store_true")
    parser.add_argument("--single-inference", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # EGL is the headless default; an explicit device selection remains
    # available through the launcher environment without assuming GPU 2.
    os.environ.setdefault("MUJOCO_GL", "egl")
    claim_output_directory(args.output_dir, managed=args.managed_output)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    config = EvaluationConfig(
        output_dir=args.output_dir.resolve(),
        policy=args.policy,
        checkpoint=args.checkpoint.resolve(),
        dataset_provenance=args.dataset_provenance,
        episodes=args.episodes,
        max_policy_steps=args.max_policy_steps,
        seed=args.seed,
        experiment=args.experiment,
        trained_run=args.trained_run,
        device=args.device,
    )
    policy = LeRobotPolicyAdapter(config.policy, config.checkpoint, config.device)
    policy.task = config.experiment
    try:
        with PickPlaceEvaluationEnvironment(config.experiment) as environment:
            runner = EvaluationRunner(config, policy, environment, stop_requested=lambda: STOP_REQUESTED)
            runner.write_metadata(policy.queue_semantics)
            if args.single_inference:
                initial = environment.reset(config.seed)
                policy.reset()
                observation = environment.observe()
                selected = policy.select_action(observation, task=config.experiment)
                environment.prepare_action(selected.postprocessed)
                report = {
                    "checkpoint_load": "PASS",
                    "observation_shapes": {key: list(value.shape) for key, value in observation.items()},
                    "observation_dtypes": {key: str(value.dtype) for key, value in observation.items()},
                    "raw_policy_output_shape": list(np.asarray(selected.raw).shape),
                    "postprocessed_action_shape": list(np.asarray(selected.postprocessed).shape),
                    "postprocessed_action_finite": bool(np.all(np.isfinite(selected.postprocessed))),
                    "postprocessed_action": np.asarray(selected.postprocessed).tolist(),
                    "initial_object_position": initial["initial_object_position"].tolist(),
                    "initial_goal_position": initial["initial_goal_position"].tolist(),
                    "robot_stepped": False,
                }
                target = config.output_dir / "single_inference.json"
                target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(report, indent=2), flush=True)
                return 0
            summary = runner.run()
            print(json.dumps(summary, indent=2), flush=True)
            return 1 if summary["system_errors"] else 0
    finally:
        # Policy owns CUDA tensors and environment owns EGL renderer state.
        del policy


if __name__ == "__main__":
    raise SystemExit(main())
