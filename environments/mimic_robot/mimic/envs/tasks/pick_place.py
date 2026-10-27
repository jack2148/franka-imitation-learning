"""Pick-and-place manipulation environment with a Franka Panda arm."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mimic.config.models import EnvConfig
from mimic.envs.base import MimicEnv
from mimic.envs.registry import register

SCENE_XML = Path(__file__).parent.parent / "assets" / "scenes" / "tabletop_pick_place.xml"

# Table surface parameters for randomizing positions
_TABLE_CENTER_X = 0.5
_TABLE_CENTER_Y = 0.0
_TABLE_Z = 0.45  # table surface (0.425) + half cube height (0.025)
_SPAWN_RANGE_X = 0.15  # +/- from table center
_SPAWN_RANGE_Y = 0.20  # +/- from table center
_MIN_CUBE_TARGET_DIST = 0.10  # minimum distance between cube and target

# Cube freejoint qpos offset (7 arm joints + 2 gripper joints = index 9)
_CUBE_QPOS_ADDR = 9

# Success threshold (meters)
_SUCCESS_THRESHOLD = 0.05


@register("pick-place")
class PickPlaceEnv(MimicEnv):
    """Pick up a cube and place it at a target location.

    The robot must pick up a red cube from the table surface and
    place it at a green target marker position.

    Observation keys:
        state: full qpos + qvel
        joint_pos: joint positions
        joint_vel: joint velocities
        image.front: front camera RGB image (480x640x3)
        image.wrist: wrist camera RGB image (480x640x3)

    Action space:
        9-dim: 7 arm joint position targets + 2 gripper finger targets

    Reward:
        Negative Euclidean distance from cube center to target center.

    Success:
        Cube center within 0.05m of target center.
    """

    def __init__(self, config: EnvConfig):
        super().__init__(config, SCENE_XML)
        self._target_pos = np.array([_TABLE_CENTER_X, _TABLE_CENTER_Y, _TABLE_Z])
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._rng = np.random.RandomState(42)

    def _reset_task(self):
        """Randomize cube start and target positions on the table."""
        # Randomize cube position on table surface
        cube_x = _TABLE_CENTER_X + self._rng.uniform(-_SPAWN_RANGE_X, _SPAWN_RANGE_X)
        cube_y = _TABLE_CENTER_Y + self._rng.uniform(-_SPAWN_RANGE_Y, _SPAWN_RANGE_Y)

        # Randomize target position ensuring minimum distance from cube
        for _ in range(100):
            target_x = _TABLE_CENTER_X + self._rng.uniform(-_SPAWN_RANGE_X, _SPAWN_RANGE_X)
            target_y = _TABLE_CENTER_Y + self._rng.uniform(-_SPAWN_RANGE_Y, _SPAWN_RANGE_Y)
            dist = np.sqrt((target_x - cube_x) ** 2 + (target_y - cube_y) ** 2)
            if dist >= _MIN_CUBE_TARGET_DIST:
                break

        # Set cube position via freejoint qpos (x, y, z, qw, qx, qy, qz)
        self.data.qpos[_CUBE_QPOS_ADDR : _CUBE_QPOS_ADDR + 3] = [cube_x, cube_y, _TABLE_Z]
        self.data.qpos[_CUBE_QPOS_ADDR + 3 : _CUBE_QPOS_ADDR + 7] = [1.0, 0.0, 0.0, 0.0]

        # Zero out cube velocity
        # freejoint dof addr is 9, 6 DOF (3 translational + 3 rotational)
        self.data.qvel[9:15] = 0.0

        # Update target position
        self._target_pos = np.array([target_x, target_y, _TABLE_Z])

        # Move the visual target marker
        self.model.body_pos[self._target_body_id] = self._target_pos.copy()

    def _get_cube_pos(self) -> np.ndarray:
        """Get current cube center position from qpos."""
        return self.data.qpos[_CUBE_QPOS_ADDR : _CUBE_QPOS_ADDR + 3].copy()

    def _compute_reward(self) -> float:
        """Dense reward: negative distance from cube to target."""
        cube_pos = self._get_cube_pos()
        dist = np.linalg.norm(cube_pos - self._target_pos)
        return -float(dist)

    def _is_success(self) -> bool:
        """Success if cube center is within threshold of target."""
        cube_pos = self._get_cube_pos()
        dist = np.linalg.norm(cube_pos - self._target_pos)
        return bool(dist < _SUCCESS_THRESHOLD)
