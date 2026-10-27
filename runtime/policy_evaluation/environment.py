"""Environment boundary for autonomous FR3 task evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_ROOT = PROJECT_ROOT / "environments" / "mimic_robot"
if str(MIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMIC_ROOT))

from mimic.config.models import CameraConfig, EnvConfig
from mimic.envs.adapter import OmyFr3MimicAdapter, TeleopCommand
from mimic.envs.tasks.pick_place_fr3 import SCENE_XML, Fr3PickPlaceEnv

from runtime.live_runner import CONTROL_DT, CONTROL_HZ, DATASET_HZ, STEPS_PER_DATASET_SAMPLE
from runtime.recording import DatasetCameraRenderer
from runtime.robot_observation import ARM_JOINT_NAMES, Fr3ObservationLayout


class InvalidActionError(ValueError):
    """A postprocessed action is unsafe or cannot represent a physical command."""


@dataclass(frozen=True)
class StepDiagnostics:
    qactual: np.ndarray
    qtarget: np.ndarray
    object_position: np.ndarray
    ee_position: np.ndarray
    robot_table_contact: bool
    object_contact: bool
    minimum_joint_limit_margin: float
    maximum_qtarget_qactual_gap: float
    clipped_dimensions: int
    maximum_action_clip: float


class PickPlaceEvaluationEnvironment:
    """Exact task reset, camera, actuator, and success semantics."""

    image_size = (480, 640)
    preview_image_size = (960, 1280)
    physics_hz = CONTROL_HZ
    policy_hz = DATASET_HZ
    physics_steps_per_action = STEPS_PER_DATASET_SAMPLE

    def __init__(self, experiment: str = "pick_and_place") -> None:
        self.experiment = experiment
        square = experiment == "mimicgen_square"
        if experiment not in {"pick_and_place", "mimicgen_square"}:
            raise ValueError(f"unsupported evaluation environment: {experiment}")
        config = EnvConfig(
            name="fr3-mimicgen-square" if square else "fr3-pick-place",
            robot="fr3",
            cameras=[
                CameraConfig(name="operator" if square else "front", width=640, height=480),
                CameraConfig(name="wrist", width=640, height=480),
            ],
            control_hz=CONTROL_HZ,
            physics_hz=CONTROL_HZ,
            episode_length=15 * CONTROL_HZ,
            action_space="joint",
        )
        if square:
            from environments.mimicgen_square.task import Fr3MimicGenSquareEnv, SCENE_XML as square_scene
            self.env = Fr3MimicGenSquareEnv(config)
            self._scene_xml = square_scene
            self._camera_names = ("operator", "wrist")
            self._observation_keys = ("observation.images.operator", "observation.images.wrist", "observation.state")
            self._object_body_name = "square_nut"
            self._object_joint_name = "square_nut_joint"
        else:
            self.env = Fr3PickPlaceEnv(config)
            self._scene_xml = SCENE_XML
            self._camera_names = ("front", "wrist")
            self._observation_keys = ("observation.images.front", "observation.images.wrist", "observation.state")
            self._object_body_name = "cube"
            self._object_joint_name = "cube_joint"
        self.model = self.env.model
        self.data = self.env.data
        if not np.isclose(self.model.opt.timestep, CONTROL_DT, atol=1e-12):
            raise RuntimeError(f"MuJoCo timestep must remain 0.001 s, got {self.model.opt.timestep}")
        if not np.isclose(self.model.body("table").pos[0], 0.595, atol=1e-12):
            raise RuntimeError("evaluation requires the validated table X=0.595 m geometry")

        self.observation_layout = Fr3ObservationLayout.from_model(self.model)
        self.action_adapter = OmyFr3MimicAdapter(self.model)
        self.renderer = DatasetCameraRenderer(
            self.model,
            image_size=self.image_size,
            cameras=self._camera_names,
            camera_overrides=getattr(self.env, "camera_overrides", None),
        )
        # Keep policy inputs at the trained 640x480 contract, while the EVAL
        # monitor uses the same large front-view framebuffer as WORK preview.
        self.preview_renderer = DatasetCameraRenderer(
            self.model,
            image_size=self.preview_image_size,
            cameras=("operator",),
            camera_overrides=getattr(self.env, "camera_overrides", None),
        )
        self.cube_body_id = self.model.body(self._object_body_name).id
        self.table_body_id = self.model.body("table").id
        self.cube_joint_id = self.model.joint(self._object_joint_name).id
        self.cube_qpos_address = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.ee_site_id = self.model.site("fr3_gripper_center_site").id
        robot_root = self.model.body("base").id
        self.robot_body_ids = {
            body_id
            for body_id in range(self.model.nbody)
            if self._is_descendant(body_id, robot_root)
        }
        self.arm_joint_ids = np.asarray([self.model.joint(name).id for name in ARM_JOINT_NAMES], dtype=int)
        self.arm_ranges = self.model.jnt_range[self.arm_joint_ids].copy()
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()
        self.policy_step = 0
        self.physics_step = 0

    @property
    def xml_path(self) -> Path:
        return self._scene_xml

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        if self.experiment == "mimicgen_square":
            self.env.reset_state(seed=seed)
        else:
            self.env._rng = np.random.RandomState(seed)
            self.env.reset_state()
        self.policy_step = 0
        self.physics_step = 0
        return {
            "initial_object_position": self.object_position,
            "initial_goal_position": self.goal_position,
        }

    @property
    def object_position(self) -> np.ndarray:
        address = self.cube_qpos_address
        return self.data.qpos[address : address + 3].copy()

    @property
    def goal_position(self) -> np.ndarray:
        return self.env.target_pos

    @property
    def simulation_time(self) -> float:
        return float(self.data.time)

    @property
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.observation_layout.arm_qpos].copy()

    def observe(self) -> dict[str, np.ndarray]:
        images = self.renderer.render(self.data)
        observation = {
            self._observation_keys[0]: images[self._camera_names[0]],
            self._observation_keys[1]: images["wrist"],
            # LeRobot training metadata declares the numeric state as float32.
            "observation.state": self.observation_layout.state(self.data).astype(np.float32),
        }
        self._validate_observation(observation, self._observation_keys)
        return observation

    def preview_image(self) -> np.ndarray:
        """Return the high-resolution oblique front/operator view for EVAL."""
        return self.preview_renderer.render(self.data)["operator"]

    @staticmethod
    def _validate_observation(observation: dict[str, np.ndarray], keys: tuple[str, ...]) -> None:
        if tuple(observation) != keys:
            raise ValueError(f"invalid policy observation keys/order: {tuple(observation)}")
        for name in keys[:2]:
            image = observation[name]
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 HWC (480, 640, 3), got {image.shape} {image.dtype}")
        state = observation["observation.state"]
        if state.shape != (16,) or state.dtype != np.float32 or not np.all(np.isfinite(state)):
            raise ValueError(f"observation.state must be finite float32 (16,), got {state.shape} {state.dtype}")

    def prepare_action(self, postprocessed_action: np.ndarray) -> tuple[np.ndarray, int, float]:
        values = np.asarray(postprocessed_action, dtype=np.float64)
        if values.shape != (8,):
            raise InvalidActionError(f"postprocessed action must have shape (8,), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise InvalidActionError("postprocessed action contains NaN or Inf")

        # Reject obviously normalized/misscaled outputs before canonical clipping.
        arm_tolerance = 0.5
        if np.any(values[:7] < self.ctrl_min[:7] - arm_tolerance) or np.any(
            values[:7] > self.ctrl_max[:7] + arm_tolerance
        ):
            raise InvalidActionError(f"arm action is implausibly outside FR3 ranges: {values[:7].tolist()}")
        if values[7] < -64.0 or values[7] > 319.0:
            raise InvalidActionError(f"gripper action is not on the expected native 0..255 scale: {values[7]}")

        applied = self.action_adapter.convert(
            TeleopCommand(joint_position=values[:7], gripper_command=float(values[7]))
        )
        clip_delta = np.abs(applied - values)
        return applied, int(np.count_nonzero(clip_delta > 1e-12)), float(np.max(clip_delta))

    def hold_action(self, postprocessed_action: np.ndarray) -> StepDiagnostics:
        applied, clipped_dimensions, maximum_action_clip = self.prepare_action(postprocessed_action)
        self.env.control(applied)
        robot_table_contact = False
        object_contact = False
        minimum_joint_limit_margin = float("inf")
        maximum_gap = 0.0
        for _ in range(self.physics_steps_per_action):
            self.env.step_physics()
            self.physics_step += 1
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                raise FloatingPointError("MuJoCo state became non-finite")
            robot_table_contact |= self._contact_between(self.robot_body_ids, {self.table_body_id})
            object_contact |= self._body_has_contact(self.cube_body_id)
            qactual = self.data.qpos[self.observation_layout.arm_qpos]
            margin = np.minimum(qactual - self.arm_ranges[:, 0], self.arm_ranges[:, 1] - qactual)
            minimum_joint_limit_margin = min(minimum_joint_limit_margin, float(np.min(margin)))
            maximum_gap = max(maximum_gap, float(np.max(np.abs(applied[:7] - qactual))))
        self.policy_step += 1
        return StepDiagnostics(
            qactual=self.data.qpos[self.observation_layout.arm_qpos].copy(),
            qtarget=applied.copy(),
            object_position=self.object_position,
            ee_position=self.data.site_xpos[self.ee_site_id].copy(),
            robot_table_contact=robot_table_contact,
            object_contact=object_contact,
            minimum_joint_limit_margin=minimum_joint_limit_margin,
            maximum_qtarget_qactual_gap=maximum_gap,
            clipped_dimensions=clipped_dimensions,
            maximum_action_clip=maximum_action_clip,
        )

    def is_success(self) -> bool:
        return bool(self.env._is_success())

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        current = body_id
        while current > 0:
            if current == ancestor_id:
                return True
            current = int(self.model.body_parentid[current])
        return current == ancestor_id

    def _body_has_contact(self, body_id: int) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if body_id in (body_a, body_b):
                return True
        return False

    def _contact_between(self, left: set[int], right: set[int]) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if (body_a in left and body_b in right) or (body_b in left and body_a in right):
                return True
        return False

    def close(self) -> None:
        self.preview_renderer.close()
        self.renderer.close()
        self.env.close()

    def __enter__(self) -> "PickPlaceEvaluationEnvironment":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
