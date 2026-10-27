"""Configuration-driven world to task-camera projection for live overlays."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import yaml

from ..paths import workspace_root


@lru_cache(maxsize=1)
def _config() -> dict[str, Any]:
    path = workspace_root() / "configs/cameras/fr3_push_dual_camera_v2.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("camera config must be a mapping")
    return value


def _camera() -> dict[str, Any]:
    camera = _config().get("task_camera")
    if not isinstance(camera, dict):
        raise ValueError("task_camera config is unavailable")
    return camera


def _rotation_from_quaternion(quaternion: list[float]) -> tuple[tuple[float, ...], ...]:
    w, x, y, z = quaternion
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def project_world_to_task_camera(point_xyz: list[float] | tuple[float, ...], *, normalized: bool = True) -> tuple[float, float, bool]:
    """Project a world/task-frame point using versioned camera extrinsics/FOV."""
    camera = _camera()
    position = [float(value) for value in camera["position"]]
    rotation = _rotation_from_quaternion([float(value) for value in camera["quaternion"]])
    delta = [float(point_xyz[index]) - position[index] for index in range(3)]
    local = [sum(rotation[row][column] * delta[row] for row in range(3)) for column in range(3)]
    depth = -local[2]
    width, height = (int(value) for value in camera.get("resolution_calibration", [448, 448]))
    tan_y = math.tan(math.radians(float(camera["fovy_deg"])) / 2.0)
    aspect = width / height
    u = width / 2.0 + local[0] / max(depth, 1e-9) * width / (2.0 * tan_y * aspect)
    v = height / 2.0 - local[1] / max(depth, 1e-9) * height / (2.0 * tan_y)
    return (u / width, v / height, depth > 1e-6) if normalized else (u, v, depth > 1e-6)


def task_region_overlay(object_pose: list[float] | None, goal_pose: list[float] | None, active_region: str | None) -> dict[str, Any]:
    """Return projected Region v1 polygons and ground-truth markers."""
    regions_path = workspace_root() / "configs/regions/fr3_push_regions_v1.yaml"
    regions = yaml.safe_load(regions_path.read_text(encoding="utf-8"))
    camera_config = yaml.safe_load((workspace_root() / "configs/cameras/fr3_push_dual_camera_v2.yaml").read_text(encoding="utf-8"))
    roi = camera_config["workspace_roi"]
    region_bounds = camera_config["object_spawn_bounds"]
    # Region A/B/C covers the same valid initial-state range used by the
    # MuJoCo classifier, not the larger visual/table ROI.
    x_min, x_max = float(region_bounds["x_min"]), float(region_bounds["x_max"])
    z_table = float(roi["z_table"])
    y_min, y_max = float(regions["lateral_min"]), float(regions["lateral_max"])
    intervals = (("A", y_min, float(regions["boundary_A_B"])), ("B", float(regions["boundary_A_B"]), float(regions["boundary_B_C"])), ("C", float(regions["boundary_B_C"]), y_max))
    polygons = []
    for label, low, high in intervals:
        world = ((x_min, low, z_table), (x_max, low, z_table), (x_max, high, z_table), (x_min, high, z_table))
        polygons.append({"id": label, "label": label, "polygon": [[u, v] for u, v, _ in (project_world_to_task_camera(point) for point in world)], "active": label == active_region})

    def marker(pose: list[float] | None) -> dict[str, Any] | None:
        if not pose or len(pose) < 3:
            return None
        u, v, visible = project_world_to_task_camera(pose[:3])
        return {"pixel": [u, v], "visible": visible}

    camera = _camera()
    resolution = camera.get("resolution_calibration", [448, 448])
    return {"camera": {"name": camera.get("name", "task_camera_v1"), "width": int(resolution[0]), "height": int(resolution[1])}, "region_version": regions.get("version"), "regions": polygons, "object": marker(object_pose), "goal": marker(goal_pose)}
