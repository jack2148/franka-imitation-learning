"""Canonical FR3 robot-state observation used by recording and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("fr3_finger_joint1", "fr3_finger_joint2")


@dataclass(frozen=True)
class Fr3ObservationLayout:
    """Named indices for the exact ``qpos7, qvel7, fingers2`` state layout."""

    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    finger_qpos: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "Fr3ObservationLayout":
        arm_joint_ids = np.asarray([model.joint(name).id for name in ARM_JOINT_NAMES], dtype=int)
        finger_joint_ids = np.asarray([model.joint(name).id for name in FINGER_JOINT_NAMES], dtype=int)
        return cls(
            arm_qpos=model.jnt_qposadr[arm_joint_ids].astype(int, copy=True),
            arm_qvel=model.jnt_dofadr[arm_joint_ids].astype(int, copy=True),
            finger_qpos=model.jnt_qposadr[finger_joint_ids].astype(int, copy=True),
        )

    def components(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            data.qpos[self.arm_qpos].copy(),
            data.qvel[self.arm_qvel].copy(),
            data.qpos[self.finger_qpos].copy(),
        )

    def state(self, data: mujoco.MjData) -> np.ndarray:
        arm_qpos, arm_qvel, finger_qpos = self.components(data)
        state = np.concatenate((arm_qpos, arm_qvel, finger_qpos))
        if state.shape != (16,):
            raise RuntimeError(f"FR3 observation state must be 16D, got {state.shape}")
        return state
