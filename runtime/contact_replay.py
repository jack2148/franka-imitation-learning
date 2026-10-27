"""Exact-state 20 Hz ZOH contact replay metrics and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .live_runner import CONTROL_DT, STEPS_PER_DATASET_SAMPLE, MimicLiveRunner


PILOT_CRITERIA = {
    "ee_position_rmse_m": 0.005,
    "final_cube_position_difference_m": 0.010,
    "contact_onset_difference_s": 0.100,
}


def _quaternion_error_deg(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(reference * candidate, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def _contact_stats(contact: np.ndarray) -> tuple[float | None, float]:
    indices = np.flatnonzero(contact)
    first = float(indices[0] * CONTROL_DT) if indices.size else None
    return first, float(indices.size * CONTROL_DT)


def replay_contact_episode(
    runner: MimicLiveRunner,
    *,
    actions_20hz: np.ndarray,
    original_arm_qpos: np.ndarray,
    original_ee_pose: np.ndarray,
    original_cube_pose: np.ndarray,
    original_contact: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Replay without invoking live OMY/controller generation."""
    runner.restore_physics(runner.initial_state)
    replay_arm: list[np.ndarray] = []
    replay_ee: list[np.ndarray] = []
    replay_cube: list[np.ndarray] = []
    replay_contact: list[bool] = []
    for step in range(original_arm_qpos.shape[0]):
        action_index = min(
            step // STEPS_PER_DATASET_SAMPLE, actions_20hz.shape[0] - 1
        )
        runner.env.control(actions_20hz[action_index])
        runner.env.step_physics()
        state = runner.post_step_state()
        replay_arm.append(state["arm_qpos"])
        replay_ee.append(state["ee_pose"])
        replay_cube.append(state["object_pose"])
        replay_contact.append(bool(state["contact"]))

    replay = {
        "arm_qpos": np.stack(replay_arm),
        "ee_pose": np.stack(replay_ee),
        "cube_pose": np.stack(replay_cube),
        "contact": np.asarray(replay_contact, dtype=np.bool_),
    }
    joint_error = replay["arm_qpos"] - original_arm_qpos
    ee_position_error = np.linalg.norm(
        replay["ee_pose"][:, 4:] - original_ee_pose[:, 4:], axis=1
    )
    ee_orientation_error = _quaternion_error_deg(
        original_ee_pose[:, :4], replay["ee_pose"][:, :4]
    )
    cube_position_error = np.linalg.norm(
        replay["cube_pose"][:, :3] - original_cube_pose[:, :3], axis=1
    )
    cube_orientation_error = _quaternion_error_deg(
        original_cube_pose[:, 3:7], replay["cube_pose"][:, 3:7]
    )
    original_onset, original_duration = _contact_stats(original_contact)
    replay_onset, replay_duration = _contact_stats(replay["contact"])
    onset_difference = (
        abs(replay_onset - original_onset)
        if replay_onset is not None and original_onset is not None
        else None
    )
    final_cube_difference = float(
        np.linalg.norm(replay["cube_pose"][-1, :3] - original_cube_pose[-1, :3])
    )
    metrics: dict[str, Any] = {
        "joint_trajectory_rmse_rad": float(np.sqrt(np.mean(joint_error**2))),
        "maximum_joint_error_rad": float(np.max(np.abs(joint_error))),
        "ee_position_rmse_m": float(np.sqrt(np.mean(ee_position_error**2))),
        "maximum_ee_position_error_m": float(np.max(ee_position_error)),
        "ee_orientation_rmse_deg": float(np.sqrt(np.mean(ee_orientation_error**2))),
        "cube_position_trajectory_rmse_m": float(np.sqrt(np.mean(cube_position_error**2))),
        "maximum_cube_position_error_m": float(np.max(cube_position_error)),
        "cube_orientation_rmse_deg": float(np.sqrt(np.mean(cube_orientation_error**2))),
        "maximum_cube_orientation_error_deg": float(np.max(cube_orientation_error)),
        "final_cube_position_difference_m": final_cube_difference,
        "original_first_contact_time_s": original_onset,
        "replay_first_contact_time_s": replay_onset,
        "contact_onset_difference_s": onset_difference,
        "original_contact_duration_s": original_duration,
        "replay_contact_duration_s": replay_duration,
        "contact_duration_difference_s": abs(replay_duration - original_duration),
        "pilot_criteria": PILOT_CRITERIA,
    }
    criteria = {
        "ee_trajectory": metrics["ee_position_rmse_m"] < PILOT_CRITERIA["ee_position_rmse_m"],
        "final_cube": final_cube_difference < PILOT_CRITERIA["final_cube_position_difference_m"],
        "contact_onset": onset_difference is not None
        and onset_difference < PILOT_CRITERIA["contact_onset_difference_s"],
        "original_contact_present": original_onset is not None,
        "replay_contact_present": replay_onset is not None,
    }
    metrics["criteria_pass"] = criteria
    metrics["pilot_pass"] = bool(all(criteria.values()))
    replay["joint_l2_error"] = np.linalg.norm(joint_error, axis=1)
    replay["ee_position_error"] = ee_position_error
    replay["cube_position_error"] = cube_position_error
    return metrics, replay


def write_contact_plots(
    output_dir: Path,
    *,
    original_arm_qpos: np.ndarray,
    original_ee_pose: np.ndarray,
    original_cube_pose: np.ndarray,
    original_contact: np.ndarray,
    replay: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            # Plotting is diagnostic output, not part of the canonical Mimic
            # dataset. Do not fail an otherwise valid episode because the
            # runtime control environment has no plotting dependency.
            return
        raise
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(original_arm_qpos.shape[0]) * CONTROL_DT
    figure, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for joint in (0, 2, 4, 6):
        axes[0].plot(t, original_arm_qpos[:, joint], label=f"q{joint + 1} original")
        axes[0].plot(t, replay["arm_qpos"][:, joint], "--", label=f"q{joint + 1} replay")
    for axis, name in enumerate("xyz"):
        axes[1].plot(t, original_ee_pose[:, 4 + axis], label=f"EE {name} original")
        axes[1].plot(t, replay["ee_pose"][:, 4 + axis], "--", label=f"EE {name} replay")
        axes[2].plot(t, original_cube_pose[:, axis], label=f"cube {name} original")
        axes[2].plot(t, replay["cube_pose"][:, axis], "--", label=f"cube {name} replay")
    axes[3].step(t, original_contact.astype(int), where="post", label="original contact")
    axes[3].step(t, replay["contact"].astype(int), where="post", label="replay contact")
    axes[0].set_ylabel("selected q [rad]")
    axes[1].set_ylabel("EE XYZ [m]")
    axes[2].set_ylabel("cube XYZ [m]")
    axes[3].set_ylabel("contact")
    axes[3].set_xlabel("simulation time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "contact_trajectory_comparison.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, replay["joint_l2_error"])
    axes[1].plot(t, replay["ee_position_error"])
    axes[2].plot(t, replay["cube_position_error"])
    axes[0].set_ylabel("joint L2 [rad]")
    axes[1].set_ylabel("EE error [m]")
    axes[2].set_ylabel("cube error [m]")
    axes[2].set_xlabel("simulation time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "contact_replay_error.png", dpi=150)
    plt.close(figure)


def save_contact_replay(
    output_dir: Path,
    metrics: dict[str, Any],
    replay: dict[str, np.ndarray],
) -> None:
    (output_dir / "contact_replay_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez(
        output_dir / "contact_replay_1000hz.npz",
        arm_qpos=replay["arm_qpos"],
        ee_pose=replay["ee_pose"],
        cube_pose=replay["cube_pose"],
        contact=replay["contact"],
    )
