"""Mimic pick-and-place task using the copied Menagerie Franka FR3."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mimic.config.models import EnvConfig
from mimic.envs.base import MimicEnv
from mimic.envs.registry import register

SCENE_XML = Path(__file__).parent.parent / "assets" / "scenes" / "tabletop_pick_place_fr3.xml"

_TABLE_CENTER_X = 0.5
_TABLE_CENTER_Y = 0.0
_TABLE_Z = 0.45
_SPAWN_RANGE_X = 0.15
_SPAWN_RANGE_Y = 0.20
_MIN_CUBE_TARGET_DIST = 0.10
_CUBE_QPOS_ADDR = 9
_SUCCESS_THRESHOLD = 0.05


@register("fr3-pick-place")
class Fr3PickPlaceEnv(MimicEnv):
    """One FR3, one cube, and the upstream Mimic pick-place task semantics.

    The 8D action is seven absolute FR3 joint-position targets followed by the
    copied FR3 gripper command in its native [0, 255] range.
    """

    def __init__(self, config: EnvConfig):
        config.robot = "fr3"
        # This local FR3 task is a 1 kHz control environment. Dataset/policy
        # decimation is owned by the live runner, not env.step().
        config.control_hz = 1000
        config.physics_hz = 1000
        if config.episode_length == 300:
            config.episode_length = 15_000  # preserve the former 15 s horizon
        super().__init__(config, SCENE_XML)
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        self._target_pos = np.array([_TABLE_CENTER_X, _TABLE_CENTER_Y, _TABLE_Z])
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._rng = np.random.RandomState(42)

    def reset(self) -> dict[str, np.ndarray]:
        self.reset_state()
        return self._get_obs()

    def reset_state(self) -> None:
        """Reset to the FR3 home keyframe without rendering cameras."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        self._step_count = 0
        self._reset_task()
        mujoco.mj_forward(self.model, self.data)

    def _reset_task(self):
        cube_x = _TABLE_CENTER_X + self._rng.uniform(-_SPAWN_RANGE_X, _SPAWN_RANGE_X)
        cube_y = _TABLE_CENTER_Y + self._rng.uniform(-_SPAWN_RANGE_Y, _SPAWN_RANGE_Y)

        for _ in range(100):
            target_x = _TABLE_CENTER_X + self._rng.uniform(-_SPAWN_RANGE_X, _SPAWN_RANGE_X)
            target_y = _TABLE_CENTER_Y + self._rng.uniform(-_SPAWN_RANGE_Y, _SPAWN_RANGE_Y)
            if np.hypot(target_x - cube_x, target_y - cube_y) >= _MIN_CUBE_TARGET_DIST:
                break

        self.data.qpos[_CUBE_QPOS_ADDR : _CUBE_QPOS_ADDR + 3] = [cube_x, cube_y, _TABLE_Z]
        self.data.qpos[_CUBE_QPOS_ADDR + 3 : _CUBE_QPOS_ADDR + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[9:15] = 0.0
        self._target_pos = np.array([target_x, target_y, _TABLE_Z])
        self.model.body_pos[self._target_body_id] = self._target_pos.copy()

    def _get_cube_pos(self) -> np.ndarray:
        return self.data.qpos[_CUBE_QPOS_ADDR : _CUBE_QPOS_ADDR + 3].copy()

    def _compute_reward(self) -> float:
        return -float(np.linalg.norm(self._get_cube_pos() - self._target_pos))

    def _is_success(self) -> bool:
        return bool(np.linalg.norm(self._get_cube_pos() - self._target_pos) < _SUCCESS_THRESHOLD)

    @property
    def target_pos(self) -> np.ndarray:
        return self._target_pos.copy()
