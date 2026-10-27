"""1000 Hz OMY-L100 -> FR3 Mimic control surface.

The timing-critical call is :meth:`MimicLiveRunner.step_once`: it obtains the
latest OMY state, executes the migrated controller, applies one 8D absolute
position command, and advances MuJoCo exactly once. Rendering and file I/O are
deliberately absent from this module's control step.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import mujoco
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = PROJECT_ROOT / 'environments' / 'mimic_robot'
if str(MIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMIC_ROOT))
from mimic.config.models import CameraConfig, EnvConfig
from mimic.envs.adapter import OmyFr3MimicAdapter, TeleopCommand
from mimic.envs.tasks.pick_place_fr3 import SCENE_XML as PICK_PLACE_SCENE_XML
from mimic.envs.tasks.pick_place_fr3 import Fr3PickPlaceEnv
from .teleop_backend import TeleopBackend, TeleopMode
CONTROL_HZ = 1000
CONTROL_DT = 1.0 / CONTROL_HZ
DATASET_HZ = 20
STEPS_PER_DATASET_SAMPLE = CONTROL_HZ // DATASET_HZ
ACTION_NAMES = tuple([f'fr3_joint{i}_target' for i in range(1, 8)] + ['fr3_gripper'])
SUPPORTED_TASKS = ('fr3-pick-place', 'mimicgen_square')
NO_OBJECT_POSE = np.zeros(0, dtype=np.float64)
NO_GOAL_POSITION = np.zeros(0, dtype=np.float64)

def _create_environment(task: str, image_size: tuple[int, int], seed: int, distribution_id: str | None=None) -> tuple[Any, Path, str]:
    aliases = {'fr3-pick-place': 'fr3-pick-place', 'pick_and_place': 'fr3-pick-place', 'mimicgen_square': 'mimicgen_square', 'fr3-mimicgen-square': 'mimicgen_square'}
    try:
        canonical_task = aliases[task]
    except KeyError as error:
        raise ValueError(f'unsupported task {task!r}; expected one of {SUPPORTED_TASKS}') from error
    config = EnvConfig(name=canonical_task, robot='fr3', cameras=[CameraConfig(name='operator', width=image_size[1], height=image_size[0]), CameraConfig(name='wrist', width=image_size[1], height=image_size[0]), CameraConfig(name='front_operator', width=image_size[1], height=image_size[0]), CameraConfig(name='left_side', width=image_size[1], height=image_size[0])], control_hz=CONTROL_HZ, physics_hz=CONTROL_HZ, episode_length=15 * CONTROL_HZ, action_space='joint')
    if canonical_task == 'mimicgen_square':
        from environments.mimicgen_square.task import SCENE_XML as SQUARE_SCENE_XML, Fr3MimicGenSquareEnv
        kwargs = {'distribution_id': distribution_id} if distribution_id else {}
        env = Fr3MimicGenSquareEnv(config, seed=seed, **kwargs)
        env.reset_state(seed=seed)
        return (env, SQUARE_SCENE_XML, canonical_task)
    env = Fr3PickPlaceEnv(config)
    env._rng = np.random.RandomState(seed)
    env.reset_state()
    return (env, PICK_PLACE_SCENE_XML, canonical_task)

@dataclass(frozen=True)
class PhysicsSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    qacc_warmstart: np.ndarray
    time: float

@dataclass(frozen=True)
class ControlObservation:
    control_step: int
    dataset_frame_index: int
    sim_time: float
    observation_state: np.ndarray
    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    gripper_qpos: np.ndarray
    action: np.ndarray
    ee_pose: np.ndarray
    object_pose: np.ndarray
    goal_position: np.ndarray
    success: bool
    physics: PhysicsSnapshot

    def numeric(self) -> dict[str, np.ndarray]:
        return {'control_step': np.asarray(self.control_step, dtype=np.int64), 'dataset_frame_index': np.asarray(self.dataset_frame_index, dtype=np.int64), 'sim_time': np.asarray(self.sim_time, dtype=np.float64), 'observation_state': self.observation_state, 'arm_qpos': self.arm_qpos, 'arm_qvel': self.arm_qvel, 'gripper_qpos': self.gripper_qpos, 'action': self.action, 'ee_pose': self.ee_pose, 'object_pose': self.object_pose, 'goal_position': self.goal_position, 'success': np.asarray(self.success, dtype=np.bool_)}

def _site_pose(data: mujoco.MjData, site_id: int) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return np.concatenate((quaternion, data.site_xpos[site_id].copy()))

class MimicLiveRunner:
    """Own the one authoritative Mimic FR3 model/data and teleop controller."""

    def __init__(self, *, input_backend: str='zmq', omy_endpoint: str='tcp://127.0.0.1:5557', seed: int=42, image_size: tuple[int, int]=(480, 640), freeze_fr3_motion: bool=False, task: str='fr3-pick-place', distribution_id: str | None=None, haptic_sender: Any=None) -> None:
        if CONTROL_HZ % DATASET_HZ:
            raise RuntimeError('control rate must be divisible by dataset rate')
        (self.env, self.xml_path, self.task) = _create_environment(task, image_size, seed, distribution_id)
        self.model = self.env.model
        self.data = self.env.data
        if not np.isclose(self.model.opt.timestep, CONTROL_DT, atol=1e-12):
            raise RuntimeError(f'MuJoCo timestep is {self.model.opt.timestep}, expected {CONTROL_DT}')
        if self.env._steps_per_control != 1:
            raise RuntimeError('live environment must advance one physics step per control tick')
        self.adapter = OmyFr3MimicAdapter(self.model)
        if input_backend == 'geomagic':
            from geomagic.leader.teleop_backend import GeomagicTeleopBackend
            self.backend = GeomagicTeleopBackend(self.model, self.data, freeze_fr3_motion=freeze_fr3_motion)
        else:
            self.backend = TeleopBackend(PROJECT_ROOT / 'teleop' / 'omy_fr3', self.model, self.data, input_backend=input_backend, teleop_mode=TeleopMode.POSITION, omy_endpoint=omy_endpoint, freeze_fr3_motion=freeze_fr3_motion)
        self.backend.initialize()
        self.backend.reset_controller_state()
        self.ee_site_id = self.model.site('fr3_gripper_center_site').id
        object_joint_name = getattr(self.env, 'object_joint_name', 'cube_joint')
        object_body_name = getattr(self.env, 'object_body_name', 'cube')
        self.has_task_object = object_joint_name is not None and object_body_name is not None
        if self.has_task_object:
            self.object_joint_id = self.model.joint(object_joint_name).id
            self.object_qpos_address = int(self.model.jnt_qposadr[self.object_joint_id])
            self.object_body_id = self.model.body(object_body_name).id
        else:
            self.object_joint_id = None
            self.object_qpos_address = None
            self.object_body_id = None
        self.target_body_name = getattr(self.env, 'target_body_name', 'target')
        self.camera_overrides = getattr(self.env, 'camera_overrides', None)
        self.cube_joint_id = self.object_joint_id
        self.cube_qpos_address = self.object_qpos_address
        self.cube_body_id = self.object_body_id
        robot_root = self.model.body('base').id
        self.robot_body_ids = {body_id for body_id in range(self.model.nbody) if self._is_descendant(body_id, robot_root)}
        self._initial_state = self.capture_physics()
        goal = self.env.target_pos
        self._initial_goal = NO_GOAL_POSITION if goal is None else goal
        self._initial_ee_pose = _site_pose(self.data, self.ee_site_id)
        self.control_step = 0
        self.haptics = None
        if haptic_sender is not None:
            from geomagic.standalone.ui_haptics import UIContactHaptics
            self.haptics = UIContactHaptics(self.backend, self.data, haptic_sender)

    @property
    def input_fresh(self) -> bool:
        return bool(self.backend.input_fresh)

    @property
    def teleop_active(self) -> bool:
        return bool(self.backend.teleop_active)

    @property
    def initial_state(self) -> PhysicsSnapshot:
        return self._initial_state

    @property
    def initial_goal(self) -> np.ndarray:
        return self._initial_goal.copy()

    @property
    def initial_ee_pose(self) -> np.ndarray:
        return self._initial_ee_pose.copy()

    def capture_physics(self) -> PhysicsSnapshot:
        return PhysicsSnapshot(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), act=self.data.act.copy(), ctrl=self.data.ctrl.copy(), qacc_warmstart=self.data.qacc_warmstart.copy(), time=float(self.data.time))

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        current = body_id
        while current > 0:
            if current == ancestor_id:
                return True
            current = int(self.model.body_parentid[current])
        return current == ancestor_id

    def object_pose_in(self, qpos: np.ndarray) -> np.ndarray:
        """Read the 7D task-object pose out of a qpos snapshot.

        Returns an empty array for a task-free environment, which has no
        object address to slice at.
        """
        if not self.has_task_object:
            return NO_OBJECT_POSE
        address = self.object_qpos_address
        return np.asarray(qpos[address:address + 7]).copy()

    def current_object_pose(self) -> np.ndarray:
        """Return the 7D task-object pose, or an empty array when there is none."""
        return self.object_pose_in(self.data.qpos)

    def current_goal_position(self) -> np.ndarray:
        """Return the task goal position, or an empty array when there is none."""
        goal = self.env.target_pos
        return NO_GOAL_POSITION if goal is None else goal

    def object_robot_contact(self) -> bool:
        """Return true only for a current FR3-body/task-object contact pair."""
        if not self.has_task_object:
            return False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if body_a == self.cube_body_id and body_b in self.robot_body_ids or (body_b == self.cube_body_id and body_a in self.robot_body_ids):
                return True
        return False

    def cube_robot_contact(self) -> bool:
        """Backward-compatible alias for stored contact diagnostics."""
        return self.object_robot_contact()

    def robot_table_contact(self) -> bool:
        """Return true for any current FR3-body/table contact pair."""
        table_body_id = self.model.body('table').id
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if body_a == table_body_id and body_b in self.robot_body_ids or (body_b == table_body_id and body_a in self.robot_body_ids):
                return True
        return False

    def restore_physics(self, state: PhysicsSnapshot) -> None:
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if self.data.act.size:
            self.data.act[:] = state.act
        self.data.ctrl[:] = state.ctrl
        self.data.qacc_warmstart[:] = state.qacc_warmstart
        self.data.time = state.time
        if self.target_body_name is not None:
            self.model.body_pos[self.model.body(self.target_body_name).id] = self._initial_goal
        if hasattr(self.env, '_target_pos'):
            self.env._target_pos = self._initial_goal.copy()
        mujoco.mj_forward(self.model, self.data)

    def reset_to_initial(self) -> None:
        if self.haptics is not None:
            self.haptics.sender.send((0, 0, 0), enabled=False)
        self.restore_physics(self._initial_state)
        self.backend.reset_controller_state()
        self.control_step = 0

    def set_teleop_mode(self, mode: TeleopMode | str) -> bool:
        return self.backend.set_teleop_mode(mode)

    def command_for_current_observation(self) -> np.ndarray:
        """Compute and apply the 8D command associated with the current state."""
        self.backend.update_command(CONTROL_DT)
        self.backend.apply_controller(self.model, self.data, CONTROL_DT)
        action = self.adapter.convert(TeleopCommand(joint_position=self.backend.hold_q_target, gripper_command=self.backend.frozen_gripper_command if self.backend.freeze_fr3_motion else self.backend.gripper_command))
        self.env.control(action)
        return action

    def observe(self, action: np.ndarray) -> ControlObservation:
        arm_qpos = self.data.qpos[self.backend.fr3_qpos_indices].copy()
        arm_qvel = self.data.qvel[self.backend.fr3_dof_indices].copy()
        gripper_qpos = self.data.qpos[7:9].copy()
        return ControlObservation(control_step=self.control_step, dataset_frame_index=self.control_step // STEPS_PER_DATASET_SAMPLE, sim_time=float(self.data.time), observation_state=np.concatenate((arm_qpos, arm_qvel, gripper_qpos)), arm_qpos=arm_qpos, arm_qvel=arm_qvel, gripper_qpos=gripper_qpos, action=np.asarray(action, dtype=np.float64).copy(), ee_pose=_site_pose(self.data, self.ee_site_id), object_pose=self.current_object_pose(), goal_position=self.current_goal_position(), success=bool(self.env._is_success()), physics=self.capture_physics())

    def step_once(self, *, capture_observation: bool=False, before_step=None) -> tuple[np.ndarray, ControlObservation | None]:
        """Execute one command and exactly one ``mj_step`` at dt=0.001 s."""
        action = self.command_for_current_observation()
        observation = self.observe(action) if capture_observation else None
        if before_step is not None and observation is not None:
            before_step(observation)
        if self.backend.freeze_fr3_motion:
            frozen_qpos_indices = np.concatenate((self.backend.fr3_qpos_indices, np.array([7, 8], dtype=int)))
            frozen_qvel_indices = np.concatenate((self.backend.fr3_dof_indices, np.array([7, 8], dtype=int)))
            frozen_qpos = self.data.qpos[frozen_qpos_indices].copy()
            frozen_qvel = self.data.qvel[frozen_qvel_indices].copy()
        tracking = getattr(self.backend, 'tracking', None)
        if tracking:
            env_start = tracking.clock()
        self.env.step_physics()
        if tracking:
            env_step_s = tracking.clock() - env_start
        if self.backend.freeze_fr3_motion:
            self.data.qpos[frozen_qpos_indices] = frozen_qpos
            self.data.qvel[frozen_qvel_indices] = frozen_qvel
            mujoco.mj_forward(self.model, self.data)
        if getattr(self.backend, 'profile', None):
            self.backend.profile.response(self.data.qpos[self.backend.fr3_qpos_indices])
        if getattr(self.backend, 'sync', None):
            self.backend.sync.tick(self.backend)
        self.control_step += 1
        if tracking:
            self.backend.finish_diagnostic_tick(env_step_s)
        if self.haptics is not None:
            self.haptics.step()
        return (action, observation)

    def post_step_state(self, *, detailed_diagnostics=True) -> dict[str, np.ndarray]:
        state = {'arm_qpos': self.data.qpos[self.backend.fr3_qpos_indices].copy(), 'ee_pose': _site_pose(self.data, self.ee_site_id), 'object_pose': self.current_object_pose(), 'contact': np.asarray(self.object_robot_contact(), dtype=np.bool_), 'robot_table_contact': np.asarray(self.robot_table_contact(), dtype=np.bool_)}
        state.update(self.backend.diagnostic_state(detailed=False) if not detailed_diagnostics and getattr(self.backend, 'input_backend', None) == 'geomagic' else self.backend.diagnostic_state())
        return state

    def state_for_ui(self) -> dict[str, Any]:
        state = self.backend.get_state()
        object_position = self.current_object_pose()[:3]
        goal_position = self.current_goal_position()
        error = float(np.linalg.norm(object_position - goal_position)) if self.has_task_object and goal_position.size else None
        result = {'sim_time': float(self.data.time), 'task': self.task, 'teleop_mode': state.mode, 'mode_change_events': list(self.backend.mode_change_events), 'input_fresh': self.input_fresh, 'teleop_active': self.teleop_active, 'clutch_count': int(state.clutch_count), 'ee_position': state.actual_position.tolist(), 'object_position': object_position.tolist(), 'goal_position': goal_position.tolist(), 'goal_error': error, 'success': bool(self.env._is_success())}
        if hasattr(self.env, 'task_diagnostics'):
            result['task_diagnostics'] = self.env.task_diagnostics()
        return result

    def input_metrics(self) -> dict[str, int | float | None]:
        return self.backend.input_metrics()

    def close(self) -> None:
        self.stop_haptics()
        self.backend.shutdown()
        self.env.close()

    def stop_haptics(self) -> None:
        if self.haptics is not None:
            self.haptics.close()

    def __enter__(self) -> 'MimicLiveRunner':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
