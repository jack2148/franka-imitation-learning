"""Source-derived geometry and reset constants for the Square D0 port."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


TASK_NAME = "mimicgen_square"
ENVIRONMENT_NAME = "fr3-mimicgen-square"

# Square_D0 is expressed around robosuite's table top at z=0.82 m. The task is
# rotated +90 degrees around Z so its long transfer axis lies laterally across
# the reachable FR3 workspace, then its origin is mapped to the current table.
UPSTREAM_TABLE_TOP = np.array([0.0, 0.0, 0.82], dtype=float)
FR3_TABLE_TOP = np.array([0.595, 0.0, 0.425], dtype=float)
UPSTREAM_TO_FR3_ROTATION = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
)
UPSTREAM_TO_FR3_TRANSLATION = FR3_TABLE_TOP - UPSTREAM_TABLE_TOP


def transform_upstream_position(position: np.ndarray) -> np.ndarray:
    return UPSTREAM_TO_FR3_ROTATION @ np.asarray(position) + UPSTREAM_TO_FR3_TRANSLATION

UPSTREAM_NUT_X_RANGE = (-0.115, -0.11)
UPSTREAM_NUT_Y_RANGE = (0.11, 0.225)
UPSTREAM_NUT_YAW_RANGE = (0.0, 2.0 * np.pi)
UPSTREAM_NUT_Z_OFFSET = 0.02
UPSTREAM_NUT_BOTTOM_OFFSET_Z = -0.05

NUT_YAW_RANGE = UPSTREAM_NUT_YAW_RANGE


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class SquareResetDistribution:
    """Bounds in the upstream Square object frame, before the +90° FR3 map."""

    distribution_id: str
    upstream_x: tuple[float, float]
    upstream_y: tuple[float, float]
    yaw_center: float
    yaw_half_range: float

    def yaw_offset(self, canonical_yaw: float) -> float:
        """Wrapped canonical yaw relative to the distribution center."""
        return wrap_to_pi(canonical_yaw - self.yaw_center)

    def contains_world_xy(self, x: float, y: float, *, tolerance: float = 1e-9) -> bool:
        (x_min, x_max), (y_min, y_max) = self.world_xy_bounds()
        return (x_min - tolerance <= x <= x_max + tolerance
                and y_min - tolerance <= y <= y_max + tolerance)

    def contains_yaw(self, canonical_yaw: float, *, tolerance: float = 1e-9) -> bool:
        return abs(self.yaw_offset(canonical_yaw)) <= self.yaw_half_range + tolerance

    def world_xy_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        corners = [
            transform_upstream_position(np.array([x, y, UPSTREAM_TABLE_TOP[2]]))
            for x in self.upstream_x for y in self.upstream_y
        ]
        return (
            (min(p[0] for p in corners), max(p[0] for p in corners)),
            (min(p[1] for p in corners), max(p[1] for p in corners)),
        )


SQUARE_UNIFORM_EVAL_V1 = SquareResetDistribution(
    "square_uniform_eval_v1", UPSTREAM_NUT_X_RANGE, UPSTREAM_NUT_Y_RANGE,
    np.pi, np.pi,
)
# Canonical yaw zero is the upstream object's yaw zero (handle along the
# upstream +X transfer axis, hole square to the axis-aligned peg). The task's
# +90° coordinate rotation makes that world yaw +pi/2, not a visually chosen
# angle. Upstream Y [0.137, 0.197] maps to FR3 X [0.398, 0.458]; upstream X
# keeps the full Square_D0 range, i.e. FR3 Y [-0.115, -0.110].
SQUARE_RESTRICTED30_V1 = SquareResetDistribution(
    "square_restricted30_v1", UPSTREAM_NUT_X_RANGE, (0.137, 0.197),
    0.0, np.pi / 4.0,
)
SQUARE_RESET_DISTRIBUTIONS = {
    item.distribution_id: item
    for item in (SQUARE_UNIFORM_EVAL_V1, SQUARE_RESTRICTED30_V1)
}
NUT_INITIAL_Z = float(
    FR3_TABLE_TOP[2] + UPSTREAM_NUT_Z_OFFSET - UPSTREAM_NUT_BOTTOM_OFFSET_Z
)

UPSTREAM_PEG_POSITION = np.array([0.23, 0.1, 0.85], dtype=float)
PEG_POSITION = transform_upstream_position(UPSTREAM_PEG_POSITION)
PEG_HALF_SIZE = np.array([0.016, 0.016, 0.1], dtype=float)

SUCCESS_XY_AXIS_TOLERANCE = 0.03
SUCCESS_MAX_NUT_Z = float(FR3_TABLE_TOP[2] + 0.05)
# Upstream requires 1 - tanh(10 * ||eef - nut||) < 0.6, i.e. the gripper
# must have moved away after release.
SUCCESS_MIN_EE_DISTANCE = float(np.arctanh(0.4) / 10.0)

# Source XML facts used by compatibility and reset-validity checks.
NUT_HORIZONTAL_RADIUS = 0.11
NUT_PHYSICAL_X_EXTENT = (-0.04375, 0.07925)
NUT_PHYSICAL_Y_EXTENT = (-0.04375, 0.04375)
NUT_HALF_THICKNESS = 0.01
NUT_HANDLE_HALF_WIDTH = 0.015875
NUT_HANDLE_FULL_WIDTH = 2.0 * NUT_HANDLE_HALF_WIDTH
FR3_MAX_FINGER_OPENING = 0.08

# The shared wrist camera points straight through the gripper. For Square, the
# handle grasp and peg become occluded by the hand at insertion height. This
# task-local side mount and pitch keep both targets visible across the tested
# grasp, peg-approach, and insertion poses.
SQUARE_WRIST_CAMERA_POS = (0.07, -0.10, 0.0)
SQUARE_WRIST_CAMERA_QUAT = (0.26693358, 0.96371493, 0.0, 0.0)
SQUARE_WRIST_CAMERA_FOVY = 90.0

TABLE_X_BOUNDS = (0.595 - 0.45, 0.595 + 0.45)
TABLE_Y_BOUNDS = (-0.55, 0.55)

# Human UI advice only. Relative pose reconstructed from the measured pre-approach
# reference and commanded EE pose in the +36 deg / -5 mm controlled condition.
# The handle breaks the hole's 90-degree symmetry, so this grasp declares none.
from runtime.grasp_guidance import GraspGuidanceConfig, Pose, SafetyConstraints

SQUARE_GRASP_GUIDANCE = GraspGuidanceConfig(
    reference_type="site", reference_name="square_nut_handle_site",
    desired_relative_pose=Pose((0.,0.,.031),
        (.012059054762100743,.9214874298129736,.3876930808896946,.020237858710353156)),
    translation_tolerance_mm=(5.,5.,2.5), orientation_tolerance_deg=5.,
    near_scale=2., ready_exit_scale=1.1, yaw_symmetry_deg=None,
    safety=SafetyConstraints((("finger_table",.5),), ("finger_table","robot_table")),
    table_geom_name="table_top", axis_labels=("Handle axis","Across handle","Height"),
    source="controlled diagnostic: offset 0 mm / yaw +36 deg / height -5 mm; one tested condition",
)
