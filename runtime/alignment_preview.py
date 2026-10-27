"""UI-only robot masking and a same-frame EE projection overlay."""

from __future__ import annotations

import math

import mujoco
import numpy as np

ALIGNMENT_CAMERA = "front_operator"
MARKER_RGB = (0, 255, 255)


def project_world_point(
    model: mujoco.MjModel, data: mujoco.MjData, camera_id: int,
    point: np.ndarray, size: tuple[int, int],
) -> np.ndarray:
    """Project a 3D point with the active MuJoCo camera (same path as EE marker)."""
    height, width = size
    local = data.cam_xmat[camera_id].reshape(3, 3).T @ (point - data.cam_xpos[camera_id])
    if model.cam_projection[camera_id] == mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC:
        scale = height / model.cam_fovy[camera_id]
    else:
        scale = height / (2 * math.tan(math.radians(model.cam_fovy[camera_id]) / 2) * -local[2])
    return np.array([width / 2 + scale * local[0], height / 2 - scale * local[1]])


def collection_boundary_projection(
    model: mujoco.MjModel, data: mujoco.MjData,
    world_xy_bounds: tuple[tuple[float, float], tuple[float, float]],
    size: tuple[int, int],
    *, camera: str = "operator",
) -> dict:
    """Project the configured XY reset rectangle onto the operator tabletop."""
    (x_min, x_max), (y_min, y_max) = world_xy_bounds
    table_id = model.geom("table_top").id
    table_top = data.geom_xpos[table_id] + (
        data.geom_xmat[table_id].reshape(3, 3)
        @ np.array([0.0, 0.0, model.geom_size[table_id, 2]])
    )
    corners = ((x_min, y_min), (x_max, y_min),
               (x_max, y_max), (x_min, y_max))
    height, width = size
    points = [
        project_world_point(model, data, model.camera(camera).id,
                            np.array([x, y, table_top[2] + 0.001]), size)
        for x, y in corners
    ]
    return {
        "camera": camera, "image_size": [width, height],
        "world_xy_bounds": world_xy_bounds,
        "polygon": [[float(p[0] / width), float(p[1] / height)] for p in points],
    }


class TopDownAlignmentPreview:
    """Render a clean workspace without changing other views or physics."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.camera_id = model.camera(ALIGNMENT_CAMERA).id
        self.ee_site_id = model.site("fr3_gripper_center_site").id
        self.table_geom_id = model.geom("table_top").id
        robot_bodies = np.zeros(model.nbody, dtype=bool)
        for body in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            robot_bodies[body] = name.startswith("fr3_") or robot_bodies[model.body_parentid[body]]
        self.robot_geoms = np.flatnonzero(robot_bodies[model.geom_bodyid])
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[5] = 0

    def update_scene(self, renderer: mujoco.Renderer, data: mujoco.MjData) -> None:
        # Temporarily reserve a disabled group only while building this scene.
        # Restore every group before rendering another pane. This model lives
        # in the UI process; the dataset writer and control model are separate.
        groups = self.model.geom_group[self.robot_geoms].copy()
        try:
            self.model.geom_group[self.robot_geoms] = 5
            renderer.update_scene(data, camera=ALIGNMENT_CAMERA, scene_option=self.scene_option)
        finally:
            self.model.geom_group[self.robot_geoms] = groups

    def marker_pixel(self, data: mujoco.MjData, size: tuple[int, int]) -> np.ndarray:
        """Project the actual gripper-center XY onto the tabletop, then camera."""
        height, width = size
        table = self.table_geom_id
        top = data.geom_xpos[table] + data.geom_xmat[table].reshape(3, 3) @ [0, 0, self.model.geom_size[table, 2]]
        point = data.site_xpos[self.ee_site_id].copy()
        point[2] = top[2]
        return project_world_point(self.model, data, self.camera_id, point, size)

    def render(self, renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
        self.update_scene(renderer, data)
        image = renderer.render()
        height, width = image.shape[:2]
        point = self.marker_pixel(data, (height, width))
        if np.isfinite(point).all():
            x, y = np.rint(point).astype(int)
            radius = max(3, round(min(height, width) / 160))
            left, right = max(0, x - radius), min(width, x + radius + 1)
            top, bottom = max(0, y - radius), min(height, y + radius + 1)
            if left < right and top < bottom:
                yy, xx = np.ogrid[top:bottom, left:right]
                distance = (xx - x) ** 2 + (yy - y) ** 2
                patch = image[top:bottom, left:right]
                patch[distance <= radius ** 2] = (0, 40, 50)
                patch[distance <= (radius - 1) ** 2] = MARKER_RGB
        return image
