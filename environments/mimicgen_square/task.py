"""MimicGen Square D0 task semantics on the shared FR3 MuJoCo robot."""

from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_ROOT = PROJECT_ROOT / "environments" / "mimic_robot"
if str(MIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMIC_ROOT))

from mimic.config.models import EnvConfig
from mimic.envs.base import MimicEnv
from mimic.envs.registry import register

from .config import (
    ENVIRONMENT_NAME,
    NUT_INITIAL_Z,
    UPSTREAM_NUT_X_RANGE,
    UPSTREAM_NUT_Y_RANGE,
    NUT_YAW_RANGE,
    PEG_POSITION,
    SUCCESS_MAX_NUT_Z,
    SUCCESS_MIN_EE_DISTANCE,
    SUCCESS_XY_AXIS_TOLERANCE,
    SQUARE_WRIST_CAMERA_FOVY,
    SQUARE_WRIST_CAMERA_POS,
    SQUARE_WRIST_CAMERA_QUAT,
    TASK_NAME,
    SQUARE_GRASP_GUIDANCE,
    SQUARE_RESET_DISTRIBUTIONS,
    SQUARE_UNIFORM_EVAL_V1,
    transform_upstream_position,
)


SCENE_XML = Path(__file__).parent / "scene.xml"


@register(ENVIRONMENT_NAME)
class Fr3MimicGenSquareEnv(MimicEnv):
    """Square_D0 reset and success logic with the canonical FR3 action API."""

    task_name = TASK_NAME
    object_joint_name = "square_nut_joint"
    object_body_name = "square_nut"
    target_body_name = None
    grasp_guidance_config = SQUARE_GRASP_GUIDANCE

    def __init__(self, config: EnvConfig, *, seed: int = 42,
                 distribution_id: str = SQUARE_UNIFORM_EVAL_V1.distribution_id):
        config.name = ENVIRONMENT_NAME
        config.robot = "fr3"
        config.control_hz = 1000
        config.physics_hz = 1000
        if config.episode_length == 300:
            config.episode_length = 15_000
        super().__init__(config, SCENE_XML)

        self.camera_overrides = {
            "wrist": {
                "pos": SQUARE_WRIST_CAMERA_POS,
                "quat": SQUARE_WRIST_CAMERA_QUAT,
                "fovy": SQUARE_WRIST_CAMERA_FOVY,
            }
        }
        self._apply_camera_overrides()

        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        self._object_joint_id = self.model.joint(self.object_joint_name).id
        self._object_qpos_address = int(
            self.model.jnt_qposadr[self._object_joint_id]
        )
        self._object_dof_address = int(
            self.model.jnt_dofadr[self._object_joint_id]
        )
        self._object_body_id = self.model.body(self.object_body_name).id
        self._peg_body_id = self.model.body("square_peg").id
        self._ee_site_id = self.model.site("fr3_gripper_center_site").id
        self._rng = np.random.RandomState(seed)
        self._planned_collection = None
        self._workspace_validation = None
        if distribution_id.startswith('workspace_validation:'):
            from runtime.workspace_validation import load_validation
            _, name, point = distribution_id.split(':')
            document = load_validation(name)
            if document['workspace_status'] != 'DRAFT':
                raise ValueError('workspace validation is already frozen')
            selected = next((p for p in document['points'] if p['name']==point), None)
            if selected is None: raise ValueError('unknown workspace validation point')
            self._workspace_validation = dict(x=selected['x'],y=selected['y'],
                z=document['object_z'], yaw=document['object_yaw_world_rad'])
            distribution_id = SQUARE_UNIFORM_EVAL_V1.distribution_id
        try:
            self.reset_distribution = SQUARE_RESET_DISTRIBUTIONS[distribution_id]
        except KeyError as error:
            if not distribution_id.startswith('square_geomagic_xy50_v'):
                raise ValueError(f"unknown Square reset distribution: {distribution_id}") from error
            from runtime.collection_plan import collection_plan
            self._planned_collection = collection_plan(distribution_id)
            self._planned_collection.require_frozen()
            self.reset_distribution = self._planned_collection
        self._last_seed = int(seed)
        self._last_reset_yaw = 0.0
        self._last_canonical_yaw = 0.0

    def _apply_camera_overrides(self) -> None:
        for name, override in self.camera_overrides.items():
            camera_id = self.model.camera(name).id
            self.model.cam_pos[camera_id] = override["pos"]
            self.model.cam_quat[camera_id] = override["quat"]
            self.model.cam_fovy[camera_id] = override["fovy"]

    def seed(self, seed: int) -> None:
        self._last_seed = int(seed)
        self._rng = np.random.RandomState(self._last_seed)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.reset_state(seed=seed)
        return self.policy_observation(render_images=True)

    def reset_state(self, seed: int | None = None) -> None:
        """Reset FR3 from the shared home key and sample Square_D0 once."""
        if seed is not None:
            self.seed(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        self._step_count = 0
        self._reset_task()
        mujoco.mj_forward(self.model, self.data)

    def _reset_task(self) -> None:
        if self._workspace_validation is not None:
            point = self._workspace_validation
            yaw = point['yaw']
            address = self._object_qpos_address
            self.data.qpos[address:address+7] = [point['x'],point['y'],point['z'],
                np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
            self.data.qvel[self._object_dof_address:self._object_dof_address+6] = 0.
            self._last_reset_yaw = float(yaw)
            self._last_canonical_yaw = float(yaw-np.pi/2)
            return
        if self._planned_collection is not None:
            state = self._planned_collection.planned_state(self._last_seed)
            document = self._planned_collection.document
            yaw = document['object_yaw_world_rad']
            address = self._object_qpos_address
            self.data.qpos[address:address+7] = [state['x'], state['y'], document['object_z'],
                np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
            self.data.qvel[self._object_dof_address:self._object_dof_address+6] = 0.
            self._last_reset_yaw = float(yaw)
            self._last_canonical_yaw = float(yaw-np.pi/2)
            return
        distribution = self.reset_distribution
        upstream_x = self._rng.uniform(*distribution.upstream_x)
        upstream_y = self._rng.uniform(*distribution.upstream_y)
        # Retain the old v1 draw order/range exactly for existing seed IDs.
        if distribution.distribution_id == SQUARE_UNIFORM_EVAL_V1.distribution_id:
            upstream_yaw = self._rng.uniform(*NUT_YAW_RANGE)
        else:
            upstream_yaw = self._rng.uniform(
                distribution.yaw_center - distribution.yaw_half_range,
                distribution.yaw_center + distribution.yaw_half_range,
            ) % (2.0 * np.pi)
        position = transform_upstream_position(
            np.array([upstream_x, upstream_y, 0.82], dtype=float)
        )
        x, y = position[:2]
        yaw = upstream_yaw + np.pi / 2.0
        quaternion = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        address = self._object_qpos_address
        self.data.qpos[address : address + 3] = [x, y, NUT_INITIAL_Z]
        self.data.qpos[address + 3 : address + 7] = quaternion
        self.data.qvel[self._object_dof_address : self._object_dof_address + 6] = 0.0
        self._last_reset_yaw = float(yaw)
        self._last_canonical_yaw = float(upstream_yaw)

    @property
    def object_qpos_address(self) -> int:
        return self._object_qpos_address

    @property
    def object_body_id(self) -> int:
        return self._object_body_id

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.xpos[self._peg_body_id].copy()

    @property
    def reset_yaw(self) -> float:
        return self._last_reset_yaw

    @property
    def canonical_reset_yaw(self) -> float:
        return self._last_canonical_yaw

    def object_pose(self) -> np.ndarray:
        address = self._object_qpos_address
        return self.data.qpos[address : address + 7].copy()

    def policy_state(self) -> np.ndarray:
        arm_qpos = np.array(
            [
                self.data.qpos[int(self.model.joint(f"fr3_joint{i}").qposadr[0])]
                for i in range(1, 8)
            ]
        )
        arm_qvel = np.array(
            [
                self.data.qvel[int(self.model.joint(f"fr3_joint{i}").dofadr[0])]
                for i in range(1, 8)
            ]
        )
        fingers = np.array(
            [
                self.data.qpos[int(self.model.joint("fr3_finger_joint1").qposadr[0])],
                self.data.qpos[int(self.model.joint("fr3_finger_joint2").qposadr[0])],
            ]
        )
        return np.concatenate((arm_qpos, arm_qvel, fingers))

    def policy_observation(self, *, render_images: bool = False) -> dict[str, np.ndarray]:
        observation = {"observation.state": self.policy_state()}
        if render_images:
            observation["observation.images.operator"] = self.render("operator")
            observation["observation.images.wrist"] = self.render("wrist")
        return observation

    def apply_action(self, action: np.ndarray) -> None:
        """Apply the unchanged 8D absolute-joint plus native-gripper action."""
        self.control(action)

    def success(self) -> bool:
        return self._is_success()

    def task_diagnostics(self) -> dict[str, float | bool | list[float]]:
        nut_position = self.data.xpos[self._object_body_id].copy()
        peg_position = self.target_pos
        relative = nut_position - peg_position
        ee_distance = float(
            np.linalg.norm(self.data.site_xpos[self._ee_site_id] - nut_position)
        )
        return {
            "object_position": nut_position.tolist(),
            "peg_position": peg_position.tolist(),
            "object_to_peg": relative.tolist(),
            "alignment_error": float(np.linalg.norm(relative[:2])),
            "insertion_depth": float(SUCCESS_MAX_NUT_Z - nut_position[2]),
            "ee_distance": ee_distance,
            "success": self._is_success(),
        }

    def _compute_reward(self) -> float:
        return float(self._is_success())

    def _is_success(self) -> bool:
        nut_position = self.data.xpos[self._object_body_id]
        peg_position = self.data.xpos[self._peg_body_id]
        ee_position = self.data.site_xpos[self._ee_site_id]
        aligned = bool(
            abs(nut_position[0] - peg_position[0]) < SUCCESS_XY_AXIS_TOLERANCE
            and abs(nut_position[1] - peg_position[1]) < SUCCESS_XY_AXIS_TOLERANCE
        )
        low_enough = bool(nut_position[2] < SUCCESS_MAX_NUT_Z)
        released = bool(
            np.linalg.norm(ee_position - nut_position) > SUCCESS_MIN_EE_DISTANCE
        )
        return aligned and low_enough and released
