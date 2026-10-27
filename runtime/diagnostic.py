"""Record a simulation-only 20 Hz episode and replay it with 50-step ZOH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any

import mujoco
import numpy as np

from .live_runner import (
    ACTION_NAMES,
    CONTROL_DT,
    CONTROL_HZ,
    DATASET_HZ,
    STEPS_PER_DATASET_SAMPLE,
    ControlObservation,
    MimicLiveRunner,
    PhysicsSnapshot,
)
from .recording import AsyncCameraWriter, stack_observations


# Declared before running the diagnostic; these are pilot engineering
# tolerances, not values selected after looking at the result.
VALIDATION_LIMITS = {
    "joint_rmse_rad": 0.01,
    "joint_max_rad": 0.05,
    "ee_position_rmse_m": 0.01,
    "ee_position_max_m": 0.03,
    "ee_orientation_rmse_deg": 2.0,
    "ee_orientation_max_deg": 5.0,
    "object_position_rmse_m": 0.005,
    "final_goal_error_delta_m": 0.005,
}


@dataclass(frozen=True)
class TimingResult:
    configured_hz: int
    achieved_hz: float
    mean_period_ms: float
    p95_period_ms: float
    p99_period_ms: float
    max_period_ms: float
    mean_work_ms: float
    p95_work_ms: float
    deadline_misses: int
    steps: int
    wall_duration_s: float


@dataclass
class EpisodeResult:
    output_dir: Path
    runner: MimicLiveRunner
    observations: list[ControlObservation]
    original_arm_qpos: np.ndarray
    original_action: np.ndarray
    original_ee_pose: np.ndarray
    original_object_pose: np.ndarray
    timing: TimingResult
    camera_frames: int


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _run_episode(
    output_dir: Path,
    *,
    duration_s: float,
    ui_live_dir: Path | None,
) -> EpisodeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = MimicLiveRunner(input_backend="synthetic")
    writer = AsyncCameraWriter(
        runner.xml_path,
        goal_position=runner.initial_goal,
        episode_dir=output_dir,
        live_dir=ui_live_dir,
    )
    observations: list[ControlObservation] = []
    arm_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    ee_states: list[np.ndarray] = []
    object_states: list[np.ndarray] = []
    periods: list[float] = []
    work_times: list[float] = []
    steps = int(round(duration_s * CONTROL_HZ))
    deadline_misses = 0
    started = time.perf_counter()
    previous_tick = started
    next_deadline = started + CONTROL_DT

    try:
        for step in range(steps):
            work_started = time.perf_counter()
            capture = step % STEPS_PER_DATASET_SAMPLE == 0
            action, observation = runner.step_once(capture_observation=capture)
            if observation is not None:
                sample_index = len(observations)
                observations.append(observation)
                writer.submit(sample_index, observation.physics)
            post = runner.post_step_state()
            actions.append(action.copy())
            arm_states.append(post["arm_qpos"])
            ee_states.append(post["ee_pose"])
            object_states.append(post["object_pose"])
            work_finished = time.perf_counter()
            work_times.append(work_finished - work_started)

            remaining = next_deadline - work_finished
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                deadline_misses += 1
            now = time.perf_counter()
            periods.append(now - previous_tick)
            previous_tick = now
            next_deadline += CONTROL_DT
            if now - next_deadline > 5.0 * CONTROL_DT:
                next_deadline = now + CONTROL_DT
        wall_duration = time.perf_counter() - started
    finally:
        writer.close()

    timing = TimingResult(
        configured_hz=CONTROL_HZ,
        achieved_hz=steps / wall_duration,
        mean_period_ms=1000.0 * statistics.fmean(periods),
        p95_period_ms=1000.0 * _percentile(periods, 95),
        p99_period_ms=1000.0 * _percentile(periods, 99),
        max_period_ms=1000.0 * max(periods),
        mean_work_ms=1000.0 * statistics.fmean(work_times),
        p95_work_ms=1000.0 * _percentile(work_times, 95),
        deadline_misses=deadline_misses,
        steps=steps,
        wall_duration_s=wall_duration,
    )
    return EpisodeResult(
        output_dir=output_dir,
        runner=runner,
        observations=observations,
        original_arm_qpos=np.stack(arm_states),
        original_action=np.stack(actions),
        original_ee_pose=np.stack(ee_states),
        original_object_pose=np.stack(object_states),
        timing=timing,
        camera_frames=writer.written,
    )


def _save_initial_state(path: Path, state: PhysicsSnapshot, goal: np.ndarray) -> None:
    np.savez(
        path,
        qpos=state.qpos,
        qvel=state.qvel,
        act=state.act,
        ctrl=state.ctrl,
        qacc_warmstart=state.qacc_warmstart,
        time=np.asarray(state.time),
        goal_position=goal,
    )


def _orientation_error_deg(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(reference[:, :4] * candidate[:, :4], axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def _replay(result: EpisodeResult) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    runner = result.runner
    runner.restore_physics(runner.initial_state)
    actions = np.stack([sample.action for sample in result.observations])
    replay_arm: list[np.ndarray] = []
    replay_ee: list[np.ndarray] = []
    replay_object: list[np.ndarray] = []
    for step in range(result.original_arm_qpos.shape[0]):
        sample = min(step // STEPS_PER_DATASET_SAMPLE, actions.shape[0] - 1)
        runner.env.control(actions[sample])
        runner.env.step_physics()
        state = runner.post_step_state()
        replay_arm.append(state["arm_qpos"])
        replay_ee.append(state["ee_pose"])
        replay_object.append(state["object_pose"])

    replay = {
        "arm_qpos": np.stack(replay_arm),
        "ee_pose": np.stack(replay_ee),
        "object_pose": np.stack(replay_object),
    }
    joint_absolute = np.abs(replay["arm_qpos"] - result.original_arm_qpos)
    ee_position_error = np.linalg.norm(
        replay["ee_pose"][:, 4:] - result.original_ee_pose[:, 4:], axis=1
    )
    ee_orientation_error = _orientation_error_deg(result.original_ee_pose, replay["ee_pose"])
    object_error = np.linalg.norm(
        replay["object_pose"][:, :3] - result.original_object_pose[:, :3], axis=1
    )
    goal = runner.initial_goal
    original_goal_error = float(np.linalg.norm(result.original_object_pose[-1, :3] - goal))
    replay_goal_error = float(np.linalg.norm(replay["object_pose"][-1, :3] - goal))
    original_success = original_goal_error < 0.05
    replay_success = replay_goal_error < 0.05
    metrics: dict[str, Any] = {
        "joint_rmse_rad": float(np.sqrt(np.mean(joint_absolute**2))),
        "joint_max_rad": float(np.max(joint_absolute)),
        "ee_position_rmse_m": float(np.sqrt(np.mean(ee_position_error**2))),
        "ee_position_max_m": float(np.max(ee_position_error)),
        "ee_orientation_rmse_deg": float(np.sqrt(np.mean(ee_orientation_error**2))),
        "ee_orientation_max_deg": float(np.max(ee_orientation_error)),
        "object_position_rmse_m": float(np.sqrt(np.mean(object_error**2))),
        "object_position_max_m": float(np.max(object_error)),
        "original_final_goal_error_m": original_goal_error,
        "replay_final_goal_error_m": replay_goal_error,
        "final_goal_error_delta_m": abs(replay_goal_error - original_goal_error),
        "original_success": original_success,
        "replay_success": replay_success,
        "success_consistent": original_success == replay_success,
    }
    limits_pass = all(
        metrics[name] <= limit for name, limit in VALIDATION_LIMITS.items()
    ) and metrics["success_consistent"]
    metrics["simulation_diagnostic_pass"] = limits_pass
    metrics["validation_limits"] = VALIDATION_LIMITS
    metrics["decision"] = "NOT YET VALIDATED"
    metrics["decision_reason"] = (
        "20 Hz ZOH passed the deterministic simulation diagnostic, but no real OMY manipulation was available."
        if limits_pass
        else "20 Hz ZOH exceeded at least one predeclared simulation tolerance."
    )
    error_series = {
        "joint_l2_rad": np.linalg.norm(replay["arm_qpos"] - result.original_arm_qpos, axis=1),
        "ee_position_m": ee_position_error,
        "ee_orientation_deg": ee_orientation_error,
        "object_position_m": object_error,
    }
    return metrics, {**replay, **error_series}


def _write_plots(path: Path, series: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = np.arange(series["joint_l2_rad"].shape[0]) * CONTROL_DT
    figure, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for axis, key, label in zip(
        axes,
        ("joint_l2_rad", "ee_position_m", "ee_orientation_deg", "object_position_m"),
        ("joint L2 error [rad]", "EE position error [m]", "EE orientation error [deg]", "object error [m]"),
    ):
        axis.plot(time_axis, series[key], linewidth=1.0)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("simulation time [s]")
    figure.suptitle("20 Hz zero-order-hold replay error")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_trajectory_plots(path: Path, result: EpisodeResult, replay: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = np.arange(result.original_arm_qpos.shape[0]) * CONTROL_DT
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for joint in range(7):
        axes[0].plot(time_axis, result.original_arm_qpos[:, joint], linewidth=1.0, label=f"q{joint + 1} original")
        axes[0].plot(time_axis, replay["arm_qpos"][:, joint], "--", linewidth=0.8, label=f"q{joint + 1} replay")
    for axis_index, name in enumerate("xyz"):
        axes[1].plot(time_axis, result.original_ee_pose[:, 4 + axis_index], label=f"EE {name} original")
        axes[1].plot(time_axis, replay["ee_pose"][:, 4 + axis_index], "--", label=f"EE {name} replay")
        axes[2].plot(time_axis, result.original_object_pose[:, axis_index], label=f"object {name} original")
        axes[2].plot(time_axis, replay["object_pose"][:, axis_index], "--", label=f"object {name} replay")
    axes[0].set_ylabel("joint position [rad]")
    axes[1].set_ylabel("EE position [m]")
    axes[2].set_ylabel("object position [m]")
    axes[2].set_xlabel("simulation time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=4, fontsize=7)
    figure.suptitle("Original 1 kHz trajectory vs 20 Hz ZOH replay")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_diagnostic(root: Path, *, duration_s: float = 2.0) -> Path:
    """Run UI-off/on timing trials and preserve the UI-on episode as diagnostic."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = root / run_id
    off = _run_episode(run_root / "timing_ui_off", duration_s=duration_s, ui_live_dir=None)
    try:
        off.runner.close()
    finally:
        pass
    live_dir = Path(__file__).resolve().parents[1] / "runs" / "work" / "live"
    episode = _run_episode(run_root / "episode", duration_s=duration_s, ui_live_dir=live_dir)
    try:
        samples = stack_observations(episode.observations)
        np.savez(episode.output_dir / "samples_20hz.npz", **samples)
        np.savez(
            episode.output_dir / "original_1000hz.npz",
            action=episode.original_action,
            arm_qpos=episode.original_arm_qpos,
            ee_pose=episode.original_ee_pose,
            object_pose=episode.original_object_pose,
        )
        _save_initial_state(
            episode.output_dir / "initial_state.npz",
            episode.runner.initial_state,
            episode.runner.initial_goal,
        )
        metrics, replay = _replay(episode)
        np.savez(
            episode.output_dir / "replay_1000hz.npz",
            arm_qpos=replay["arm_qpos"],
            ee_pose=replay["ee_pose"],
            object_pose=replay["object_pose"],
        )
        _write_plots(episode.output_dir / "replay_error.png", replay)
        _write_trajectory_plots(
            episode.output_dir / "trajectory_comparison.png", episode, replay
        )
        (episode.output_dir / "replay_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        metadata = {
            "run_id": run_id,
            "source": "synthetic_omy_simulation_only",
            "real_omy_hardware_verified": False,
            "haptic": False,
            "environment": "fr3-pick-place",
            "control_hz": CONTROL_HZ,
            "mujoco_timestep_s": CONTROL_DT,
            "mj_steps_per_control_tick": 1,
            "dataset_hz": DATASET_HZ,
            "physics_steps_per_dataset_action": STEPS_PER_DATASET_SAMPLE,
            "temporal_convention": "observation_t_before_step -> action_t_applied_to_that_state",
            "action_names": ACTION_NAMES,
            "action_shape": [8],
            "action_semantics": "absolute 7D FR3 joint-position target + native scalar gripper command",
            "sample_count": len(episode.observations),
            "camera_frames": episode.camera_frames,
            "observation_state_shape": [16],
            "camera_shape": [480, 640, 3],
            "camera_hz": DATASET_HZ,
            "ui_off_timing": asdict(off.timing),
            "ui_on_timing": asdict(episode.timing),
            "ui_on_definition": "same 20 Hz dataset rendering plus atomic legacy-console JPEG publication",
            "push_only_success_possible": True,
            "push_only_reason": "success depends only on cube-to-target 3D distance < 0.05 m",
        }
        (episode.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        (run_root / "timing_comparison.json").write_text(
            json.dumps(
                {"ui_off": asdict(off.timing), "ui_on": asdict(episode.timing)},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    finally:
        episode.runner.close()
    return run_root
