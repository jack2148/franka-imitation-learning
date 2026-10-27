"""Canonical and historical visual-observation schema definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


CURRENT_CAMERA_SCHEMA = "operator_wrist_v1"
LEGACY_CAMERA_SCHEMA = "front_wrist_legacy_v1"
OPERATOR_CAMERA_CONFIG_VERSION = "operator_fixed_oblique_v1"
# The left-side monitor is an oblique lateral view from the robot's left,
# ~31 degrees above horizontal, framed on the measured teleoperation
# envelope. It shows the FR3 so the operator can read arm motion and
# approach height. UI-only: it is never stored in a learning dataset.
LEFT_SIDE_CAMERA_CONFIG_VERSION = "left_side_oblique_lateral_v1"

SCHEMA_CAMERAS = {
    CURRENT_CAMERA_SCHEMA: ("operator", "wrist"),
    LEGACY_CAMERA_SCHEMA: ("front", "wrist"),
}


def camera_names_for_schema(schema: str) -> tuple[str, ...]:
    try:
        return SCHEMA_CAMERAS[schema]
    except KeyError as error:
        raise ValueError(f"unsupported camera schema: {schema}") from error


def detect_raw_camera_schema(root: Path, metadata: dict[str, Any]) -> str:
    """Identify a raw episode without reinterpreting camera semantics."""
    declared = metadata.get("camera_schema")
    image_root = root / "images"
    populated = {
        camera
        # Dataset cameras plus every UI-only camera, so an episode that
        # wrongly stored a display view is detected rather than ignored.
        for camera in ("operator", "wrist", "front", "front_operator", "left_side", "top_down")
        if (image_root / camera).is_dir()
        and any((image_root / camera).glob("*.png"))
    }
    if declared is not None:
        schema = str(declared)
        expected = set(camera_names_for_schema(schema))
        if populated != expected:
            raise ValueError(
                f"declared camera schema {schema} expects {sorted(expected)}, "
                f"found {sorted(populated)}"
            )
        return schema
    if populated == {"operator", "wrist"}:
        return CURRENT_CAMERA_SCHEMA
    if populated == {"front", "wrist"}:
        return LEGACY_CAMERA_SCHEMA
    raise ValueError(
        "unable to identify raw camera schema; expected exactly operator+wrist "
        f"or front+wrist, found {sorted(populated)}"
    )
