"""Thin adapter around the current OMY_FRANKA_TELEOP controller functions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

import mujoco
import numpy as np


TELEOP_ENVIRONMENT_VARIABLE = "OMY_FRANKA_TELEOP_ROOT"
BRIDGE_RELATIVE_PATH = Path("launch/FR3_omy_bridge.py")
OMY_RELATIVE_PATH = Path("robotis_mujoco_menagerie/robotis_omy/scene.xml")
INPUT_BACKENDS = {"ros", "zmq", "synthetic"}


class TeleopMode(str, Enum):
    """Mutually exclusive operator-controlled Cartesian subspaces."""

    POSITION = "position"
    ROTATION = "rotation"

    @property
    def controller_mode(self) -> str:
        # Both components remain constrained by the validated 6D DLS. The
        # mode gate controls which desired-pose component the OMY may change;
        # the other component is still actively held at its frozen target.
        return "full_pose"

    @classmethod
    def parse(cls, value: "TeleopMode | str") -> "TeleopMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            raise ValueError("teleop mode must be position or rotation") from error


def resolve_teleop_root(
    cli_path: str | Path | None = None,
    *,
    study_root: Path | None = None,
) -> Path:
    """Resolve the external dependency in CLI, env, sibling order.

    An explicit source (CLI flag or environment variable) is authoritative:
    if it does not point at a valid dependency, this raises immediately
    instead of silently falling back to a sibling checkout.
    """

    def _validated(source: str, candidate: Path) -> Path | None:
        resolved = candidate.resolve()
        bridge = resolved / BRIDGE_RELATIVE_PATH
        return resolved if bridge.is_file() else None

    if cli_path is not None:
        candidate = Path(cli_path).expanduser()
        resolved = _validated("CLI --teleop-root", candidate)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            "CLI --teleop-root does not point at a valid OMY_FRANKA_TELEOP "
            f"checkout: {candidate.resolve()} (missing {BRIDGE_RELATIVE_PATH})"
        )

    environment_path = os.environ.get(TELEOP_ENVIRONMENT_VARIABLE)
    if environment_path:
        candidate = Path(environment_path).expanduser()
        resolved = _validated(TELEOP_ENVIRONMENT_VARIABLE, candidate)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            f"{TELEOP_ENVIRONMENT_VARIABLE} does not point at a valid "
            f"OMY_FRANKA_TELEOP checkout: {candidate.resolve()} "
            f"(missing {BRIDGE_RELATIVE_PATH})"
        )

    if study_root is None:
        study_root = Path(__file__).resolve().parents[3]
    attempted: list[str] = []
    for candidate in (
        study_root / "vendor" / "omy_franka_teleop",
        study_root.parent / "OMY_FRANKA_TELEOP",
        study_root.parent / "omy_franka_teleop",
    ):
        resolved = _validated("sibling", candidate)
        if resolved is not None:
            return resolved
        attempted.append(f"sibling: {candidate.resolve()}")
    details = "\n  ".join(attempted) if attempted else "(no candidates)"
    raise FileNotFoundError(
        "Could not locate OMY_FRANKA_TELEOP. Set --teleop-root or "
        f"{TELEOP_ENVIRONMENT_VARIABLE}. Tried:\n  {details}"
    )


def _load_bridge(teleop_root: Path) -> ModuleType:
    bridge_path = teleop_root / BRIDGE_RELATIVE_PATH
    module_name = "_fr3_block_push_external_bridge"
    specification = importlib.util.spec_from_file_location(module_name, bridge_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load teleop bridge: {bridge_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except ModuleNotFoundError as error:
        if error.name and (
            error.name.startswith("rclpy")
            or error.name.startswith("sensor_msgs")
            or error.name.startswith("std_msgs")
        ):
            raise RuntimeError(
                "ROS 2 Python packages required by OMY_FRANKA_TELEOP are not "
                "available. Source the matching ROS 2 environment and use its "
                "Python interpreter."
            ) from error
        raise
    return module


def _load_controller_only_bridge(teleop_root: Path) -> ModuleType:
    """Load the existing controller definitions without importing ROS.

    The external file combines ROS Node code and controller math. IPC mode
    removes only ROS imports and the ``OmyPose`` Node class from the parsed
    module; mapping, conditioning, FK helpers, and IK execute from that source.
    """
    bridge_path = teleop_root / BRIDGE_RELATIVE_PATH
    tree = ast.parse(bridge_path.read_text(), filename=str(bridge_path))
    # Load only state handling, Cartesian mapping, target conditioning, and
    # FR3 DLS IK. Haptic modules/topics remain forbidden for compatibility
    # with older source variants and are never imported or called.
    forbidden_modules = {
        "rclpy", "sensor_msgs", "std_msgs", "haptics",
        "fr3_collision_feedback",
    }
    filtered_body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name.split(".", 1)[0] in forbidden_modules
            for alias in node.names
        ):
            continue
        if isinstance(node, ast.ImportFrom) and (
            node.module or ""
        ).split(".", 1)[0] in forbidden_modules:
            continue
        if isinstance(node, ast.ClassDef) and node.name == "OmyPose":
            continue
        filtered_body.append(node)
    tree.body = filtered_body
    ast.fix_missing_locations(tree)

    module_name = "_fr3_block_push_external_controller"
    module = ModuleType(module_name)
    module.__file__ = str(bridge_path)
    sys.modules[module_name] = module
    try:
        exec(compile(tree, str(bridge_path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass(frozen=True)
class TeleopState:
    mode: str
    active: bool
    ros_fresh: bool
    clutch_count: int
    command_position: np.ndarray
    target_position: np.ndarray
    actual_position: np.ndarray


class TeleopBackend:
    """Use external mapping/conditioning/DLS while accepting final model/data."""

    def __init__(
        self,
        teleop_root: Path,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        input_backend: str = "ros",
        teleop_mode: TeleopMode | str = TeleopMode.POSITION,
        omy_endpoint: str = "tcp://127.0.0.1:5557",
        omy_stale_timeout: float = 0.2,
        freeze_fr3_motion: bool = False,
    ) -> None:
        if input_backend not in INPUT_BACKENDS:
            raise ValueError(
                f"input_backend must be one of {sorted(INPUT_BACKENDS)}"
            )
        if omy_stale_timeout <= 0:
            raise ValueError("omy_stale_timeout must be positive")
        self.teleop_root = teleop_root
        self.model = model
        self.data = data
        self.input_backend = input_backend
        self.omy_endpoint = omy_endpoint
        self.omy_stale_timeout = float(omy_stale_timeout)
        self.freeze_fr3_motion = bool(freeze_fr3_motion)
        self.bridge = (
            _load_bridge(teleop_root)
            if input_backend == "ros"
            else _load_controller_only_bridge(teleop_root)
        )
        self.teleop_mode = TeleopMode.parse(teleop_mode)
        self.mode_change_events: list[dict[str, int | float | str]] = []
        self.dt = float(model.opt.timestep)

        omy_path = teleop_root / OMY_RELATIVE_PATH
        if not omy_path.is_file():
            raise FileNotFoundError(f"OMY FK model not found: {omy_path}")
        self.omy_model = mujoco.MjModel.from_xml_path(str(omy_path))
        self.omy_data = mujoco.MjData(self.omy_model)
        mujoco.mj_resetDataKeyframe(
            self.omy_model,
            self.omy_data,
            self.bridge.keyframe_id(self.omy_model, "home"),
        )
        mujoco.mj_forward(self.omy_model, self.omy_data)

        self.omy_base_body_id = self.bridge.body_id(
            self.omy_model, self.bridge.OMY_BASE_BODY_NAME
        )
        self.omy_ee_site_id = self.bridge.site_id(
            self.omy_model, self.bridge.OMY_EE_SITE_NAME
        )
        self.omy_qpos_addresses = [
            int(self.omy_model.jnt_qposadr[self.omy_model.joint(name).id])
            for name in self.bridge.OMY_MUJOCO_JOINTS
        ]

        self.fr3_ee_site_id = self.bridge.site_id(
            model, self.bridge.FR3_EE_SITE_NAME
        )
        joint_ids = np.array(
            [model.joint(f"fr3_joint{i}").id for i in range(1, 8)],
            dtype=int,
        )
        self.fr3_qpos_indices = model.jnt_qposadr[joint_ids].astype(int)
        self.fr3_dof_indices = model.jnt_dofadr[joint_ids].astype(int)
        self.fr3_joint_lower = model.jnt_range[joint_ids, 0].copy()
        self.fr3_joint_upper = model.jnt_range[joint_ids, 1].copy()
        self.fr3_actuator_indices = np.array(
            [model.actuator(f"fr3_joint{i}").id for i in range(1, 8)],
            dtype=int,
        )
        self.fr3_gripper_actuator_id = model.actuator("fr3_gripper").id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

        self.state_lock = threading.Lock()
        self.joint_positions = self.omy_data.qpos[
            self.omy_qpos_addresses
        ].copy()
        self.ros_node = None
        self.ros_thread: threading.Thread | None = None
        self.omy_client = None
        self.gripper_command = 0.0
        self.frozen_gripper_command = float(data.ctrl[self.fr3_gripper_actuator_id])
        self.latest_trigger_position = 0.0
        self.latest_omy_joint_positions = self.joint_positions.copy()
        omy_position, omy_rotation = self.bridge.read_site_pose(
            self.omy_data, self.omy_ee_site_id
        )
        self.latest_omy_position = omy_position.copy()
        self.latest_omy_rotation = omy_rotation.copy()
        self.reset_controller_state()

    def initialize(self) -> None:
        if self.input_backend == "zmq":
            from .ipc.omy_state_client import OmyStateClient

            self.omy_client = OmyStateClient(
                self.omy_endpoint,
                stale_timeout=self.omy_stale_timeout,
            )
            return
        if self.input_backend == "synthetic":
            return
        rclpy = self.bridge.rclpy
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = self.bridge.OmyPose(
            self.joint_positions,
            self.state_lock,
        )
        self.ros_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.ros_node,),
            daemon=True,
        )
        self.ros_thread.start()

    def reset_controller_state(self) -> None:
        position, rotation = self.bridge.read_site_pose(
            self.data, self.fr3_ee_site_id
        )
        self.fr3_command_position = position.copy()
        self.fr3_command_rotation = rotation.copy()
        self.fr3_target_position = position.copy()
        self.fr3_target_rotation = rotation.copy()
        self.hold_q_target = self.data.qpos[self.fr3_qpos_indices].copy()
        self.nullspace_posture_reference = self.hold_q_target.copy()
        self.target_linear_velocity = np.zeros(3)
        self.target_angular_velocity = np.zeros(3)
        self.teleop_active = False
        self.clutch_count = 0
        self.omy_anchor_position = None
        self.omy_anchor_rotation = None
        self.fr3_anchor_position = position.copy()
        self.fr3_anchor_rotation = rotation.copy()
        self.ros_fresh = False
        self.input_fresh = False

    def set_teleop_mode(self, mode: TeleopMode | str) -> bool:
        """Switch motion subspaces with an atomic canonical re-anchor.

        The FR3 side uses the current unconditioned command, matching the
        existing clutch anchor convention. Neither command nor conditioned
        target is changed at the transition itself, so an active-clutch
        switch cannot produce an instantaneous target discontinuity.
        """
        new_mode = TeleopMode.parse(mode)
        if new_mode is self.teleop_mode:
            return False

        old_mode = self.teleop_mode
        # Refresh the leader FK at the command boundary so an active switch
        # does not reinterpret motion that arrived after the previous 1 kHz
        # control tick. A stale source safely falls back to the latest pose;
        # normal stale/recovery handling remains owned by update_command().
        omy_target, _, _, fresh = self.update_omy_state()
        if fresh:
            self._update_omy_fk(omy_target)
        self.omy_anchor_position = self.latest_omy_position.copy()
        self.omy_anchor_rotation = self.latest_omy_rotation.copy()
        self.fr3_anchor_position = self.fr3_command_position.copy()
        self.fr3_anchor_rotation = self.fr3_command_rotation.copy()
        self.target_linear_velocity.fill(0.0)
        self.target_angular_velocity.fill(0.0)
        self.teleop_mode = new_mode
        self.mode_change_events.append({
            "timestamp": float(self.data.time),
            "wall_time_unix_s": time.time(),
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
            "clutch_count": int(self.clutch_count),
        })
        return True

    def _update_omy_fk(self, omy_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for address, position in zip(self.omy_qpos_addresses, omy_target):
            self.omy_data.qpos[address] = position
        self.latest_omy_joint_positions = np.asarray(omy_target, dtype=float).copy()
        mujoco.mj_forward(self.omy_model, self.omy_data)
        omy_position, omy_rotation = self.bridge.read_site_pose(
            self.omy_data, self.omy_ee_site_id
        )
        self.latest_omy_position = omy_position.copy()
        self.latest_omy_rotation = omy_rotation.copy()
        return omy_position, omy_rotation

    def update_omy_state(self) -> tuple[np.ndarray, float, float, bool]:
        if self.input_backend == "synthetic":
            # Deterministic simulation-only source used by diagnostics. It
            # drives the unmodified OMY FK/mapping/conditioning/IK path.
            t = float(self.data.time)
            target = self.joint_positions.copy()
            motion_t = max(0.0, t - 0.20)
            envelope = min(1.0, motion_t / 0.30)
            target[0] += envelope * 0.10 * np.sin(2.0 * np.pi * 0.30 * motion_t)
            target[1] += envelope * 0.07 * np.sin(2.0 * np.pi * 0.22 * motion_t)
            target[3] += envelope * 0.08 * np.sin(2.0 * np.pi * 0.18 * motion_t)
            return target, float(self.bridge.TRIGGER_ON_THRESHOLD - 0.05), 255.0, True

        if self.input_backend == "zmq":
            if self.omy_client is None:
                raise RuntimeError("initialize() must be called before teleoperation")
            state = self.omy_client.read()
            if state.message is None:
                return (
                    self.joint_positions.copy(),
                    float(self.bridge.TRIGGER_OFF_THRESHOLD),
                    self.gripper_command,
                    False,
                )
            target = np.asarray(state.message.position, dtype=float)
            if target.shape != (6,) or not np.all(np.isfinite(target)):
                return (
                    self.joint_positions.copy(),
                    float(self.bridge.TRIGGER_OFF_THRESHOLD),
                    self.gripper_command,
                    False,
                )
            if state.fresh:
                self.joint_positions[:] = target
            trigger = float(state.message.trigger_position)
            gripper = float(state.message.gripper_command)
            return target, trigger, gripper, state.fresh

        if self.ros_node is None:
            raise RuntimeError("initialize() must be called before teleoperation")
        with self.state_lock:
            target = self.joint_positions.copy()
            trigger = float(self.ros_node.trigger_position)
            gripper = float(self.ros_node.gripper_command)
            last_message = float(self.ros_node.last_message_time)
            has_state = bool(self.ros_node.has_joint_state)
        fresh = (
            has_state
            and time.perf_counter() - last_message <= self.bridge.ROS_TIMEOUT_S
        )
        return target, trigger, gripper, fresh

    def update_command(self, dt: float) -> None:
        omy_target, trigger, self.gripper_command, fresh = self.update_omy_state()
        self.latest_trigger_position = trigger
        recovered = fresh and not self.input_fresh
        self.input_fresh = fresh
        self.ros_fresh = fresh
        if not fresh:
            self.teleop_active = False
            self.target_linear_velocity.fill(0.0)
            self.target_angular_velocity.fill(0.0)
            self.fr3_command_position = self.fr3_target_position.copy()
            self.fr3_command_rotation = self.fr3_target_rotation.copy()
            return

        if recovered:
            # The first state after startup/staleness becomes the new OMY
            # reference. This makes its initial Cartesian delta exactly zero.
            self.teleop_active = False
            self.target_linear_velocity.fill(0.0)
            self.target_angular_velocity.fill(0.0)
            self.fr3_command_position = self.fr3_target_position.copy()
            self.fr3_command_rotation = self.fr3_target_rotation.copy()

        omy_position, omy_rotation = self._update_omy_fk(omy_target)

        if (
            not self.teleop_active
            and trigger <= self.bridge.TRIGGER_ON_THRESHOLD
        ):
            self.clutch_count += 1
            self.omy_anchor_position = omy_position.copy()
            self.omy_anchor_rotation = omy_rotation.copy()
            self.fr3_anchor_position = self.fr3_command_position.copy()
            self.fr3_anchor_rotation = self.fr3_command_rotation.copy()
            self.teleop_active = True
            self.target_linear_velocity.fill(0.0)
            self.target_angular_velocity.fill(0.0)
        elif (
            self.teleop_active
            and trigger >= self.bridge.TRIGGER_OFF_THRESHOLD
        ):
            self.teleop_active = False

        if self.teleop_active:
            desired_position, desired_rotation = self.bridge.make_desired_target(
                self.omy_anchor_position,
                self.omy_anchor_rotation,
                omy_position,
                omy_rotation,
                self.fr3_anchor_position,
                self.fr3_anchor_rotation,
            )
            if self.teleop_mode is TeleopMode.POSITION:
                self.fr3_command_position = desired_position
                self.fr3_command_rotation = self.fr3_anchor_rotation.copy()
            else:
                self.fr3_command_position = self.fr3_anchor_position.copy()
                # Rotation mode is yaw-only: preserve the anchored tool X/Y
                # orientation and project the leader's relative rotation onto
                # the anchored Z axis.  The bridge returns a full 3D target,
                # so applying it directly would let roll and pitch drift.
                relative_rotation = (
                    self.fr3_anchor_rotation.T @ desired_rotation
                )
                relative_rotvec = self.bridge.matrix_to_rotvec(relative_rotation)
                yaw_only = np.array(
                    [0.0, 0.0, float(relative_rotvec[2])], dtype=float
                )
                self.fr3_command_rotation = self.fr3_anchor_rotation @ self.bridge.rotvec_to_matrix(yaw_only)

        # Freeze the inactive component at the target-generation boundary.
        # Feeding the current conditioned target back as that component's
        # desired value prevents leader coupling and hidden conditioning
        # wind-up while preserving target continuity across mode switches.
        conditioned_position = self.fr3_command_position
        conditioned_rotation = self.fr3_command_rotation
        if self.teleop_mode is TeleopMode.POSITION:
            conditioned_rotation = self.fr3_target_rotation
            self.target_angular_velocity.fill(0.0)
        else:
            conditioned_position = self.fr3_target_position
            self.target_linear_velocity.fill(0.0)

        conditioned = self.bridge.condition_target(
            self.fr3_target_position,
            self.fr3_target_rotation,
            conditioned_position,
            conditioned_rotation,
            self.target_linear_velocity,
            self.target_angular_velocity,
            dt,
        )
        self.fr3_target_position = conditioned[0]
        self.fr3_target_rotation = conditioned[1]
        self.target_linear_velocity = conditioned[5]
        self.target_angular_velocity = conditioned[6]
        if self.teleop_mode is TeleopMode.POSITION:
            self.target_angular_velocity.fill(0.0)
        else:
            self.target_linear_velocity.fill(0.0)

    def apply_controller(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt: float,
    ) -> None:
        if self.freeze_fr3_motion:
            # OMY-only calibration still updates FK diagnostics, but never
            # forwards those changes to the FR3 arm or gripper actuators.
            data.ctrl[self.fr3_actuator_indices] = self.hold_q_target
            data.ctrl[self.fr3_gripper_actuator_id] = self.frozen_gripper_command
            return
        position_gap = (
            np.linalg.norm(self.fr3_command_position - self.fr3_target_position)
            if self.teleop_mode is TeleopMode.POSITION
            else 0.0
        )
        rotation_gap = (
            np.linalg.norm(self.bridge.matrix_to_rotvec(
                self.fr3_command_rotation @ self.fr3_target_rotation.T
            ))
            if self.teleop_mode is TeleopMode.ROTATION
            else 0.0
        )
        target_follow_active = (
            self.teleop_active
            or position_gap >= 1e-6
            or rotation_gap >= 1e-4
        )
        if target_follow_active and self.ros_fresh:
            target, _ = self.bridge.compute_joint_target(
                model,
                data,
                self.fr3_ee_site_id,
                self.fr3_dof_indices,
                self.fr3_qpos_indices,
                self.fr3_joint_lower,
                self.fr3_joint_upper,
                self.fr3_target_position,
                self.fr3_target_rotation,
                self.hold_q_target,
                self.jacp,
                self.jacr,
                dt,
                teleop_mode=self.teleop_mode.controller_mode,
                enable_nullspace_posture=self.bridge.ENABLE_NULLSPACE_POSTURE,
                q_posture_reference=self.nullspace_posture_reference,
            )
            self.hold_q_target = target.copy()
        data.ctrl[self.fr3_actuator_indices] = self.hold_q_target
        data.ctrl[self.fr3_gripper_actuator_id] = self.gripper_command

    def get_state(self) -> TeleopState:
        actual, _ = self.bridge.read_site_pose(self.data, self.fr3_ee_site_id)
        return TeleopState(
            mode=self.teleop_mode.value,
            active=self.teleop_active,
            ros_fresh=self.ros_fresh,
            clutch_count=self.clutch_count,
            command_position=self.fr3_command_position.copy(),
            target_position=self.fr3_target_position.copy(),
            actual_position=actual,
        )

    def input_metrics(self) -> dict[str, int | float | None]:
        """Return transport counters without changing controller behavior."""
        if self.omy_client is None:
            return {
                "received_messages": 0,
                "dropped_sequences": 0,
                "rejected_messages": 0,
                "stale_reads": 0,
                "last_message_age_s": None,
            }
        state = self.omy_client.read()
        return {
            "received_messages": self.omy_client.received_messages,
            "dropped_sequences": self.omy_client.dropped_sequences,
            "rejected_messages": self.omy_client.rejected_messages,
            "stale_reads": self.omy_client.stale_reads,
            "last_message_age_s": state.age_seconds,
        }

    @staticmethod
    def _pose_vector(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        return np.concatenate((quaternion, position))

    def diagnostic_state(self) -> dict[str, np.ndarray]:
        """Copy live mapping state for verification; never mutates control."""
        actual_position, actual_rotation = self.bridge.read_site_pose(
            self.data, self.fr3_ee_site_id
        )
        return {
            "teleop_mode": np.asarray(self.teleop_mode.value),
            "input_fresh": np.asarray(self.input_fresh, dtype=np.bool_),
            "teleop_active": np.asarray(self.teleop_active, dtype=np.bool_),
            "clutch_count": np.asarray(self.clutch_count, dtype=np.int64),
            "trigger_position": np.asarray(
                self.latest_trigger_position, dtype=np.float64
            ),
            "gripper_command": np.asarray(self.gripper_command, dtype=np.float64),
            "omy_joint_position": self.latest_omy_joint_positions.copy(),
            "omy_ee_pose": self._pose_vector(
                self.latest_omy_position, self.latest_omy_rotation
            ),
            "command_ee_pose": self._pose_vector(
                self.fr3_command_position, self.fr3_command_rotation
            ),
            "target_ee_pose": self._pose_vector(
                self.fr3_target_position, self.fr3_target_rotation
            ),
            "actual_ee_pose": self._pose_vector(actual_position, actual_rotation),
        }

    def shutdown(self) -> None:
        if self.omy_client is not None:
            self.omy_client.close()
            self.omy_client = None
        if self.ros_node is not None:
            self.ros_node.destroy_node()
            self.ros_node = None
        if self.input_backend == "ros" and self.bridge.rclpy.ok():
            self.bridge.rclpy.shutdown()
