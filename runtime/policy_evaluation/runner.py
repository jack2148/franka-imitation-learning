"""Shared 20 Hz closed-loop evaluator and lightweight result persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from .environment import InvalidActionError, PickPlaceEvaluationEnvironment


class PolicyLike(Protocol):
    policy_name: str
    checkpoint: Path

    def reset(self) -> None: ...
    def select_action(self, observation: dict[str, np.ndarray]) -> Any: ...


@dataclass(frozen=True)
class EvaluationConfig:
    output_dir: Path
    policy: str
    checkpoint: Path
    dataset_provenance: str
    episodes: int = 1
    max_policy_steps: int = 2400
    seed: int = 0
    experiment: str = "pick_and_place"
    trained_run: str = ""
    device: str = "cuda"

    def validate(self) -> None:
        if self.experiment not in {"pick_and_place", "mimicgen_square"}:
            raise ValueError(f"unsupported evaluation environment: {self.experiment}")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be positive")
        if self.policy not in {"act", "diffusion_policy"}:
            raise ValueError(f"unsupported policy: {self.policy}")


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    initial_object_position: list[float]
    initial_goal_position: list[float]
    result: str
    policy_steps: int
    physics_steps: int
    simulated_duration: float
    success_policy_step: int | None
    success_sim_time: float | None
    termination_reason: str
    final_goal_distance: float
    robot_table_contact_steps: int
    object_contact_steps: int
    minimum_joint_limit_margin: float | None
    maximum_qtarget_qactual_gap: float
    invalid_action_count: int
    clipped_action_dimensions: int
    maximum_action_clip: float
    maximum_policy_interval_joint_delta: float
    distinct_front_frames: int
    distinct_wrist_frames: int


def claim_output_directory(path: Path, *, managed: bool = False) -> None:
    """Create a new run directory, never reusing an existing result location."""
    if managed:
        if not path.is_dir():
            raise FileNotFoundError(f"managed output directory was not created by the launcher: {path}")
        allowed = {"console.log", "evaluation_config.json"}
        unexpected = [entry.name for entry in path.iterdir() if entry.name not in allowed]
        if unexpected:
            raise FileExistsError(f"refusing non-empty managed output directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)


class EvaluationRunner:
    def __init__(
        self,
        config: EvaluationConfig,
        policy: PolicyLike,
        environment: PickPlaceEvaluationEnvironment,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.policy = policy
        self.environment = environment
        self.stop_requested = stop_requested or (lambda: False)
        self.results: list[EpisodeResult] = []
        self.logs: dict[str, list[np.ndarray | int | float | bool]] = {
            "episode_index": [],
            "policy_step": [],
            "physics_step": [],
            "simulation_time": [],
            "raw_action": [],
            "postprocessed_action": [],
            "applied_action": [],
            "qactual": [],
            "object_position": [],
            "ee_position": [],
            "robot_table_contact": [],
            "object_contact": [],
        }
        self.trajectory: list[list[float]] = []
        self.frame_seq = 0

    def write_metadata(self, queue_semantics: dict[str, Any]) -> None:
        primary_camera = "operator" if self.config.experiment == "mimicgen_square" else "front"
        metadata = {
            "experiment": self.config.experiment,
            "environment": self.config.experiment,
            "policy": self.config.policy,
            "trained_run": self.config.trained_run,
            "checkpoint": str(self.config.checkpoint.resolve()),
            "dataset_provenance": self.config.dataset_provenance,
            "episodes_requested": self.config.episodes,
            "max_policy_steps": self.config.max_policy_steps,
            "maximum_simulated_duration_seconds": self.config.max_policy_steps / 20.0,
            "base_seed": self.config.seed,
            "seeds": [self.config.seed + index for index in range(self.config.episodes)],
            "device": self.config.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mujoco_gl": os.environ.get("MUJOCO_GL", "egl"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "physics_hz": self.environment.physics_hz,
            "policy_hz": self.environment.policy_hz,
            "physics_steps_per_policy_action": self.environment.physics_steps_per_action,
            "preview": {
                "camera": "operator",
                "shape": [960, 1280, 3],
                "poll_hz": 60,
            },
            "observation": {
                f"observation.images.{primary_camera}": {"shape": [480, 640, 3], "dtype": "uint8"},
                "observation.images.wrist": {"shape": [480, 640, 3], "dtype": "uint8"},
                "observation.state": {
                    "shape": [16],
                    "dtype": "float32",
                    "order": ["FR3 qpos 7", "FR3 qvel 7", "finger qpos 2"],
                },
            },
            "action": {
                "shape": [8],
                "semantics": ["absolute FR3 joint target radians x7", "native gripper 0 closed / 255 open"],
            },
            "success_condition": "task environment success predicate",
            "table_x_m": 0.595,
            "command_to_actual_ee_tracking_error": "not available: controller commands joint targets, not EE poses",
            "queue_semantics": queue_semantics,
            "mimic_git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "lerobot_git_commit": _git_commit(Path(os.environ.get("MIMIC_LEROBOT_ROOT", str(Path(__file__).resolve().parents[2] / "vendor/lerobot")))),
            "created_at": time.time(),
        }
        _atomic_json(self.config.output_dir / "evaluation_metadata.json", metadata)

    def run(self) -> dict[str, Any]:
        self._write_state(status="RUNNING", episode=0, policy_step=0, simulation_time=0.0)
        for episode_index in range(self.config.episodes):
            if self.stop_requested():
                break
            seed = self.config.seed + episode_index
            initial = self.environment.reset(seed)
            self.policy.reset()
            self.trajectory = []
            result = self._run_episode(episode_index, seed, initial)
            self.results.append(result)
            self._write_episode_metrics()
            self._write_summary("STOPPED" if result.result == "ABORTED" else "RUNNING")
            if result.result in {"ABORTED", "SYSTEM_ERROR"}:
                break

        stopped = bool(self.results and self.results[-1].result == "ABORTED") or self.stop_requested()
        final_status = "STOPPED" if stopped else "COMPLETED"
        summary = self._write_summary(final_status)
        self._write_actions()
        self._write_state(
            status=final_status,
            episode=len(self.results),
            policy_step=self.results[-1].policy_steps if self.results else 0,
            simulation_time=self.results[-1].simulated_duration if self.results else 0.0,
            termination_reason=self.results[-1].termination_reason if self.results else final_status,
        )
        return summary

    def _run_episode(self, episode_index: int, seed: int, initial: dict[str, np.ndarray]) -> EpisodeResult:
        start_time = self.environment.simulation_time
        robot_table_contacts = 0
        object_contacts = 0
        invalid_action_count = 0
        clipped_dimensions = 0
        maximum_action_clip = 0.0
        minimum_margin = float("inf")
        maximum_gap = 0.0
        maximum_joint_delta = 0.0
        previous_qactual = self.environment.arm_qpos
        front_hashes: set[str] = set()
        wrist_hashes: set[str] = set()
        result = "TASK_FAILURE"
        reason = "maximum policy steps reached without success"
        success_step: int | None = None
        success_time: float | None = None
        steps = 0

        for policy_step in range(self.config.max_policy_steps):
            if self.stop_requested():
                result, reason = "ABORTED", "operator stop requested"
                break
            try:
                observation = self.environment.observe()
                image_keys = [key for key in observation if key.startswith("observation.images.")]
                if len(image_keys) < 2:
                    raise ValueError("evaluation requires two camera observations")
                front_hashes.add(_frame_hash(observation[image_keys[0]]))
                wrist_hashes.add(_frame_hash(observation[image_keys[1]]))
                self._write_preview(self.environment.preview_image())
                selected = self.policy.select_action(observation)
                postprocessed = np.asarray(selected.postprocessed)
                diagnostics = self.environment.hold_action(postprocessed)
            except InvalidActionError as error:
                invalid_action_count += 1
                result, reason = "SYSTEM_ERROR", f"invalid policy action: {error}"
                break
            except Exception as error:
                result, reason = "SYSTEM_ERROR", f"{type(error).__name__}: {error}"
                break

            steps = policy_step + 1
            robot_table_contacts += int(diagnostics.robot_table_contact)
            object_contacts += int(diagnostics.object_contact)
            clipped_dimensions += diagnostics.clipped_dimensions
            maximum_action_clip = max(maximum_action_clip, diagnostics.maximum_action_clip)
            minimum_margin = min(minimum_margin, diagnostics.minimum_joint_limit_margin)
            maximum_gap = max(maximum_gap, diagnostics.maximum_qtarget_qactual_gap)
            maximum_joint_delta = max(
                maximum_joint_delta, float(np.max(np.abs(diagnostics.qactual - previous_qactual)))
            )
            previous_qactual = diagnostics.qactual.copy()
            self.trajectory.append(diagnostics.ee_position[:2].tolist())
            self._append_log(episode_index, policy_step, selected.raw, postprocessed, diagnostics)
            if policy_step < 10:
                print(f"episode {episode_index + 1} action {policy_step}: {postprocessed.tolist()}", flush=True)
            self._write_state(
                status="RUNNING",
                episode=episode_index + 1,
                policy_step=steps,
                physics_step=self.environment.physics_step,
                simulation_time=self.environment.simulation_time - start_time,
                result="RUNNING",
            )
            if self.environment.is_success():
                result, reason = "SUCCESS", "current task success condition reached"
                success_step = steps
                success_time = self.environment.simulation_time - start_time
                break

        duration = self.environment.simulation_time - start_time
        goal_distance = float(np.linalg.norm(self.environment.object_position - self.environment.goal_position))
        return EpisodeResult(
            episode_index=episode_index,
            seed=seed,
            initial_object_position=np.asarray(initial["initial_object_position"]).tolist(),
            initial_goal_position=np.asarray(initial["initial_goal_position"]).tolist(),
            result=result,
            policy_steps=steps,
            physics_steps=self.environment.physics_step,
            simulated_duration=duration,
            success_policy_step=success_step,
            success_sim_time=success_time,
            termination_reason=reason,
            final_goal_distance=goal_distance,
            robot_table_contact_steps=robot_table_contacts,
            object_contact_steps=object_contacts,
            minimum_joint_limit_margin=None if minimum_margin == float("inf") else minimum_margin,
            maximum_qtarget_qactual_gap=maximum_gap,
            invalid_action_count=invalid_action_count,
            clipped_action_dimensions=clipped_dimensions,
            maximum_action_clip=maximum_action_clip,
            maximum_policy_interval_joint_delta=maximum_joint_delta,
            distinct_front_frames=len(front_hashes),
            distinct_wrist_frames=len(wrist_hashes),
        )

    def _append_log(self, episode_index: int, policy_step: int, raw: np.ndarray, final: np.ndarray, diag: Any) -> None:
        values = {
            "episode_index": episode_index,
            "policy_step": policy_step,
            "physics_step": self.environment.physics_step,
            "simulation_time": self.environment.simulation_time,
            "raw_action": np.asarray(raw).reshape(-1),
            "postprocessed_action": np.asarray(final).reshape(-1),
            "applied_action": diag.qtarget,
            "qactual": diag.qactual,
            "object_position": diag.object_position,
            "ee_position": diag.ee_position,
            "robot_table_contact": diag.robot_table_contact,
            "object_contact": diag.object_contact,
        }
        for key, value in values.items():
            self.logs[key].append(value)

    def _write_preview(self, image: np.ndarray) -> None:
        path = self.config.output_dir / "latest_frame.jpg"
        temporary = path.with_name(f".{path.name}.tmp")
        Image.fromarray(image, mode="RGB").save(temporary, format="JPEG", quality=90)
        temporary.replace(path)
        self.frame_seq += 1

    def _write_state(self, *, status: str, episode: int, policy_step: int, simulation_time: float, **extra: Any) -> None:
        body = {
            "status": status,
            "policy": self.config.policy,
            "trained_run": self.config.trained_run,
            "checkpoint": self.config.checkpoint.parent.name,
            "episode": episode,
            "total_episodes": self.config.episodes,
            "policy_step": policy_step,
            "step": policy_step,
            "physics_step": extra.pop("physics_step", policy_step * self.environment.physics_steps_per_action),
            "simulation_time": simulation_time,
            "frame_url": "/api/eval/frame",
            "frame_seq": self.frame_seq,
            "trajectory": self.trajectory[-500:],
            **extra,
        }
        _atomic_json(self.config.output_dir / "eval_state.json", body)

    def _write_episode_metrics(self) -> None:
        path = self.config.output_dir / "episode_metrics.csv"
        if not self.results:
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(self.results[0])))
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))

    def _write_summary(self, status: str) -> dict[str, Any]:
        completed = [item for item in self.results if item.result in {"SUCCESS", "TASK_FAILURE"}]
        successes = sum(item.result == "SUCCESS" for item in self.results)
        task_failures = sum(item.result == "TASK_FAILURE" for item in self.results)
        system_errors = sum(item.result == "SYSTEM_ERROR" for item in self.results)
        aborted = sum(item.result == "ABORTED" for item in self.results)
        summary = {
            "status": status,
            "policy": self.config.policy,
            "trained_run": self.config.trained_run,
            "checkpoint": str(self.config.checkpoint.resolve()),
            "dataset_provenance": self.config.dataset_provenance,
            "episodes_requested": self.config.episodes,
            "episodes": len(completed),
            "episodes_completed": len(completed),
            "task_successes": successes,
            "task_failures": task_failures,
            "system_errors": system_errors,
            "aborted_episodes": aborted,
            "success_rate": successes / len(completed) if completed else None,
            "mean_policy_steps": float(np.mean([item.policy_steps for item in completed])) if completed else None,
            "mean_simulated_duration": float(np.mean([item.simulated_duration for item in completed])) if completed else None,
            "seeds": [item.seed for item in self.results],
            "episode_results": [asdict(item) for item in self.results],
        }
        _atomic_json(self.config.output_dir / "summary.json", summary)
        return summary

    def _write_actions(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for key, values in self.logs.items():
            arrays[key] = np.asarray(values)
        temporary = self.config.output_dir / ".actions.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(self.config.output_dir / "actions.npz")


def _frame_hash(image: np.ndarray) -> str:
    return hashlib.blake2b(memoryview(np.ascontiguousarray(image)), digest_size=8).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ("git", "-C", str(path), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or "unknown"
    except OSError:
        return "unknown"
