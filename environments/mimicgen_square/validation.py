#!/usr/bin/env python3
"""Headless geometry, reset, reachability, and camera validation for Square."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_ROOT = PROJECT_ROOT / "environments" / "mimic_robot"
if str(MIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMIC_ROOT))

import mujoco
import numpy as np
from PIL import Image

from mimic.config.models import CameraConfig, EnvConfig

from .config import (
    FR3_TABLE_TOP,
    NUT_PHYSICAL_X_EXTENT,
    NUT_PHYSICAL_Y_EXTENT,
    PEG_POSITION,
    SUCCESS_MAX_NUT_Z,
    TABLE_X_BOUNDS,
    TABLE_Y_BOUNDS,
    UPSTREAM_NUT_X_RANGE,
    UPSTREAM_NUT_Y_RANGE,
)
from .task import Fr3MimicGenSquareEnv


def make_environment(image_size: tuple[int, int] = (480, 640)) -> Fr3MimicGenSquareEnv:
    return Fr3MimicGenSquareEnv(
        EnvConfig(
            name="fr3-mimicgen-square",
            robot="fr3",
            cameras=[
                CameraConfig(name="operator", width=image_size[1], height=image_size[0]),
                CameraConfig(name="wrist", width=image_size[1], height=image_size[0]),
                CameraConfig(name="front_operator", width=image_size[1], height=image_size[0]),
                CameraConfig(name="left_side", width=image_size[1], height=image_size[0]),
            ],
            control_hz=1000,
            physics_hz=1000,
            episode_length=15_000,
            action_space="joint",
        )
    )


def _descendants(model: mujoco.MjModel, root_name: str) -> set[int]:
    root = model.body(root_name).id
    result = set()
    for body_id in range(model.nbody):
        current = body_id
        while current > 0:
            if current == root:
                result.add(body_id)
                break
            current = int(model.body_parentid[current])
    return result


def _contact_exists(
    env: Fr3MimicGenSquareEnv, left: set[int], right: set[int]
) -> bool:
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        body_a = int(env.model.geom_bodyid[contact.geom1])
        body_b = int(env.model.geom_bodyid[contact.geom2])
        if (body_a in left and body_b in right) or (
            body_a in right and body_b in left
        ):
            return True
    return False


def _footprint_supported(position: np.ndarray, yaw: float) -> bool:
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    corners = np.array(
        [
            [x, y]
            for x in NUT_PHYSICAL_X_EXTENT
            for y in NUT_PHYSICAL_Y_EXTENT
        ]
    )
    world = corners @ rotation.T + position[:2]
    return bool(
        np.all((world[:, 0] >= TABLE_X_BOUNDS[0]) & (world[:, 0] <= TABLE_X_BOUNDS[1]))
        and np.all((world[:, 1] >= TABLE_Y_BOUNDS[0]) & (world[:, 1] <= TABLE_Y_BOUNDS[1]))
    )


def validate_resets(env: Fr3MimicGenSquareEnv, count: int = 100) -> dict[str, Any]:
    robot = _descendants(env.model, "base")
    table = {env.model.body("table").id}
    task_object = {env.object_body_id}
    peg = {env.model.body("square_peg").id}
    samples = []
    invalid = []
    for seed in range(count):
        env.reset_state(seed=seed)
        pose = env.object_pose()
        yaw = float(2.0 * np.arctan2(pose[6], pose[3]))
        row = [float(pose[0]), float(pose[1]), float(env.reset_yaw)]
        samples.append(row)
        reasons = []
        # +90-degree task-frame rotation maps upstream Y to -world X and
        # upstream X to world Y.
        if not 0.595 - UPSTREAM_NUT_Y_RANGE[1] <= pose[0] <= 0.595 - UPSTREAM_NUT_Y_RANGE[0]:
            reasons.append("object_x")
        if not UPSTREAM_NUT_X_RANGE[0] <= pose[1] <= UPSTREAM_NUT_X_RANGE[1]:
            reasons.append("object_y")
        if not _footprint_supported(pose[:3], yaw):
            reasons.append("object_footprint")
        if _contact_exists(env, robot, table):
            reasons.append("robot_table_contact")
        if _contact_exists(env, task_object, table | peg | robot):
            reasons.append("initial_object_contact")
        if env.success():
            reasons.append("success_at_reset")
        if reasons:
            invalid.append({"seed": seed, "reasons": reasons})
    values = np.asarray(samples)
    return {
        "seeds_tested": count,
        "invalid_resets": len(invalid),
        "invalid_details": invalid,
        "object_x_range_m": [float(values[:, 0].min()), float(values[:, 0].max())],
        "object_y_range_m": [float(values[:, 1].min()), float(values[:, 1].max())],
        "object_world_yaw_range_rad": [
            float(values[:, 2].min()),
            float(values[:, 2].max()),
        ],
        "target_position_m": PEG_POSITION.tolist(),
    }


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    if angle < 1e-8:
        return skew / 2.0
    return angle * skew / (2.0 * np.sin(angle))


def solve_home_orientation_ik(
    env: Fr3MimicGenSquareEnv, target: np.ndarray
) -> dict[str, Any]:
    model, data = env.model, env.data
    site_id = model.site("fr3_gripper_center_site").id
    qpos_ids = np.array(
        [int(model.joint(f"fr3_joint{i}").qposadr[0]) for i in range(1, 8)]
    )
    dof_ids = np.array(
        [int(model.joint(f"fr3_joint{i}").dofadr[0]) for i in range(1, 8)]
    )
    lower = np.array([model.joint(f"fr3_joint{i}").range[0] for i in range(1, 8)])
    upper = np.array([model.joint(f"fr3_joint{i}").range[1] for i in range(1, 8)])
    home = model.key_qpos[model.key("home").id, qpos_ids].copy()
    data.qpos[qpos_ids] = home
    mujoco.mj_forward(model, data)
    target_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    position_error = np.full(3, np.inf)
    rotation_error = np.full(3, np.inf)
    iterations = 0
    for iterations in range(1500):
        position_error = np.asarray(target) - data.site_xpos[site_id]
        rotation_error = _rotation_vector(
            target_rotation @ data.site_xmat[site_id].reshape(3, 3).T
        )
        if np.linalg.norm(position_error) < 2e-4 and np.linalg.norm(rotation_error) < 2e-3:
            break
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        jacobian = np.vstack((jac_pos[:, dof_ids], jac_rot[:, dof_ids]))
        error = np.concatenate((position_error, rotation_error))
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 2e-4 * np.eye(6), error
        )
        data.qpos[qpos_ids] = np.clip(
            data.qpos[qpos_ids] + 0.5 * np.clip(delta, -0.03, 0.03),
            lower + 1e-4,
            upper - 1e-4,
        )
        mujoco.mj_forward(model, data)
    position_norm = float(np.linalg.norm(position_error))
    rotation_norm = float(np.linalg.norm(rotation_error))
    qpos = data.qpos[qpos_ids].copy()
    return {
        "reachable": position_norm < 2e-4 and rotation_norm < 2e-3,
        "iterations": iterations,
        "position_error_m": position_norm,
        "rotation_error_rad": rotation_norm,
        "joint_limit_margin_rad": float(np.min(np.minimum(qpos - lower, upper - qpos))),
        "qpos": qpos.tolist(),
    }


def validate_reachability(env: Fr3MimicGenSquareEnv) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env.reset_state(seed=0)
    handle_xy = env.data.site("square_nut_handle_site").xpos[:2].copy()
    peg = env.target_pos
    peg_handle_xy = peg[:2] + np.array([0.0, 0.054])
    targets = {
        "approach": np.r_[handle_xy, FR3_TABLE_TOP[2] + 0.22],
        "grasp": np.r_[handle_xy, FR3_TABLE_TOP[2] + 0.04],
        "lift": np.r_[handle_xy, FR3_TABLE_TOP[2] + 0.22],
        "peg_approach": np.r_[peg_handle_xy, FR3_TABLE_TOP[2] + 0.23],
        "final_insertion": np.r_[peg_handle_xy, FR3_TABLE_TOP[2] + 0.04],
    }
    solutions: dict[str, np.ndarray] = {}
    results = {}
    robot = _descendants(env.model, "base")
    table = {env.model.body("table").id}
    for name, target in targets.items():
        result = solve_home_orientation_ik(env, target)
        solutions[name] = np.asarray(result["qpos"], dtype=float)
        result["target_m"] = target.tolist()
        result["robot_table_contact"] = _contact_exists(env, robot, table)
        results[name] = result
    return results, solutions


def _visible_pixels(
    env: Fr3MimicGenSquareEnv,
    camera: str,
    body_id: int,
    scene_option: mujoco.MjvOption | None = None,
) -> int:
    renderer = mujoco.Renderer(env.model, height=240, width=320)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera=camera, scene_option=scene_option)
        segmentation = renderer.render()
    finally:
        renderer.close()
    geom_ids = np.flatnonzero(env.model.geom_bodyid == body_id)
    return int(np.count_nonzero(np.isin(segmentation[:, :, 0], geom_ids)))


def validate_cameras(
    env: Fr3MimicGenSquareEnv,
    solutions: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_qpos_ids = np.array(
        [int(env.model.joint(f"fr3_joint{i}").qposadr[0]) for i in range(1, 8)]
    )
    global_counts = []
    for seed in (0, 42, 99):
        env.reset_state(seed=seed)
        global_counts.append(
            {
                "seed": seed,
                "operator_object_pixels": _visible_pixels(env, "operator", env.object_body_id),
                "operator_peg_pixels": _visible_pixels(env, "operator", env.model.body("square_peg").id),
                "left_side_object_pixels": _visible_pixels(env, "left_side", env.object_body_id),
                "left_side_peg_pixels": _visible_pixels(env, "left_side", env.model.body("square_peg").id),
            }
        )
        if seed == 0:
            for camera in ("operator", "front_operator", "left_side"):
                Image.fromarray(env.render(camera)).save(
                    output_dir / f"{camera}_seed0.png"
                )

    env.reset_state(seed=0)
    env.data.qpos[robot_qpos_ids] = solutions["grasp"]
    mujoco.mj_forward(env.model, env.data)
    wrist_object_pixels = _visible_pixels(env, "wrist", env.object_body_id)
    Image.fromarray(env.render("wrist")).save(output_dir / "wrist_grasp.png")

    env.reset_state(seed=0)
    env.data.qpos[robot_qpos_ids] = solutions["peg_approach"]
    mujoco.mj_forward(env.model, env.data)
    wrist_peg_pixels = _visible_pixels(
        env, "wrist", env.model.body("square_peg").id
    )
    Image.fromarray(env.render("wrist")).save(output_dir / "wrist_peg_approach.png")

    env.reset_state(seed=0)
    env.data.qpos[robot_qpos_ids] = solutions["final_insertion"]
    object_address = env.object_qpos_address
    env.data.qpos[object_address : object_address + 3] = [
        PEG_POSITION[0],
        PEG_POSITION[1],
        SUCCESS_MAX_NUT_Z - 0.02,
    ]
    env.data.qpos[object_address + 3 : object_address + 7] = [
        np.sqrt(0.5),
        0.0,
        0.0,
        np.sqrt(0.5),
    ]
    mujoco.mj_forward(env.model, env.data)
    wrist_peg_pixels_at_insertion = _visible_pixels(
        env, "wrist", env.model.body("square_peg").id
    )
    wrist_object_pixels_at_insertion = _visible_pixels(
        env, "wrist", env.object_body_id
    )
    Image.fromarray(env.render("wrist")).save(output_dir / "wrist_insertion.png")
    return {
        "global_views_multiple_seeds": global_counts,
        "wrist_object_pixels_at_grasp": wrist_object_pixels,
        "wrist_peg_pixels_at_approach": wrist_peg_pixels,
        "wrist_peg_pixels_at_insertion": wrist_peg_pixels_at_insertion,
        "wrist_object_pixels_at_insertion": wrist_object_pixels_at_insertion,
        "screenshots": {
            "operator": str(output_dir / "operator_seed0.png"),
            "front": str(output_dir / "front_operator_seed0.png"),
            "left_side": str(output_dir / "left_side_seed0.png"),
            "wrist_grasp": str(output_dir / "wrist_grasp.png"),
            "wrist_peg_approach": str(output_dir / "wrist_peg_approach.png"),
            "wrist_insertion": str(output_dir / "wrist_insertion.png"),
        },
    }


def run_validation(output_dir: Path) -> dict[str, Any]:
    env = make_environment()
    try:
        resets = validate_resets(env, 100)
        reachability, solutions = validate_reachability(env)
        cameras = validate_cameras(env, solutions, output_dir)
        return {
            "reset": resets,
            "reachability": reachability,
            "camera": cameras,
            "action_dim": env.action_dim,
            "state_dim": int(env.policy_state().shape[0]),
            "object_mass_kg": float(env.model.body("square_nut").mass[0]),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/mimicgen_square_validation"))
    args = parser.parse_args()
    result = run_validation(args.output_dir.resolve())
    report = args.output_dir.resolve() / "validation.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
