"""Versioned wire protocol for OMY joint positions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Sequence


PROTOCOL_VERSION = 3
OMY_JOINT_NAMES = tuple(f"Joint{i}" for i in range(1, 7))
OMY_ROS_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
OMY_TRIGGER_JOINT = "rh_r1_joint"
MESSAGE_FIELDS = {
    "protocol_version",
    "sequence",
    "source_timestamp_ns",
    "sender_monotonic_ns",
    "joint_names",
    "position",
    "trigger_position",
    "gripper_command",
}


class ProtocolError(ValueError):
    """Raised when a payload cannot be trusted as an OMY state."""


@dataclass(frozen=True)
class OmyStateMessage:
    protocol_version: int
    sequence: int
    source_timestamp_ns: int
    sender_monotonic_ns: int
    joint_names: tuple[str, ...]
    position: tuple[float, ...]
    trigger_position: float
    gripper_command: float


def _validated_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")
    return value


def _validated_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{name} contains NaN or Inf")
    return number


def extract_joint_position(
    joint_names: Sequence[str],
    position: Sequence[float],
    joint_name: str,
) -> float:
    """Extract exactly one finite named joint position."""
    if len(joint_names) != len(position):
        raise ProtocolError("joint_names and position lengths differ")
    matches = [index for index, name in enumerate(joint_names) if name == joint_name]
    if not matches:
        raise ProtocolError(f"missing required joint: {joint_name}")
    if len(matches) != 1:
        raise ProtocolError(f"duplicate required joint: {joint_name}")
    return _validated_number(
        f"position for {joint_name}", position[matches[0]]
    )


def reorder_joint_positions(
    joint_names: Sequence[str],
    position: Sequence[float],
    *,
    allow_extra_joints: bool = False,
    required_joint_names: Sequence[str] = OMY_JOINT_NAMES,
) -> tuple[float, ...]:
    """Return positions in the requested six-joint order."""
    if len(joint_names) != len(position):
        raise ProtocolError("joint_names and position lengths differ")
    if any(not isinstance(name, str) for name in joint_names):
        raise ProtocolError("all joint names must be strings")
    required = tuple(required_joint_names)
    if len(required) != 6 or len(set(required)) != 6:
        raise ProtocolError("required_joint_names must contain six unique names")

    indices: dict[str, int] = {}
    for index, name in enumerate(joint_names):
        if name in required:
            if name in indices:
                raise ProtocolError(f"duplicate required joint: {name}")
            indices[name] = index
        elif not allow_extra_joints:
            raise ProtocolError(f"unexpected joint: {name}")

    missing = [name for name in required if name not in indices]
    if missing:
        raise ProtocolError(f"missing required joints: {missing}")

    ordered: list[float] = []
    for name in required:
        raw_value = position[indices[name]]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ProtocolError(f"position for {name} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ProtocolError(f"position for {name} contains NaN or Inf")
        ordered.append(value)
    if len(ordered) != 6:
        raise ProtocolError("position must have shape (6,)")
    return tuple(ordered)


def encode_message(
    *,
    sequence: int,
    source_timestamp_ns: int,
    sender_monotonic_ns: int,
    joint_names: Sequence[str],
    position: Sequence[float],
    trigger_position: float,
    gripper_command: float,
) -> bytes:
    ordered = reorder_joint_positions(joint_names, position)
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "sequence": _validated_integer("sequence", sequence),
        "source_timestamp_ns": _validated_integer(
            "source_timestamp_ns", source_timestamp_ns
        ),
        "sender_monotonic_ns": _validated_integer(
            "sender_monotonic_ns", sender_monotonic_ns
        ),
        "joint_names": list(OMY_JOINT_NAMES),
        "position": list(ordered),
        "trigger_position": _validated_number(
            "trigger_position", trigger_position
        ),
        "gripper_command": _validated_number(
            "gripper_command", gripper_command
        ),
    }
    return json.dumps(message, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def decode_message(
    payload: bytes,
    *,
    last_sequence: int | None = None,
) -> OmyStateMessage:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("payload is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or set(decoded) != MESSAGE_FIELDS:
        raise ProtocolError("message fields do not match the protocol")
    version = _validated_integer("protocol_version", decoded["protocol_version"])
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol_version: {version}")
    sequence = _validated_integer("sequence", decoded["sequence"])
    if last_sequence is not None and sequence <= last_sequence:
        raise ProtocolError(
            f"sequence must increase: received {sequence} after {last_sequence}"
        )
    source_timestamp_ns = _validated_integer(
        "source_timestamp_ns", decoded["source_timestamp_ns"]
    )
    sender_monotonic_ns = _validated_integer(
        "sender_monotonic_ns", decoded["sender_monotonic_ns"]
    )
    names = decoded["joint_names"]
    positions = decoded["position"]
    if not isinstance(names, list) or not isinstance(positions, list):
        raise ProtocolError("joint_names and position must be arrays")
    ordered = reorder_joint_positions(names, positions)
    trigger_position = _validated_number(
        "trigger_position", decoded["trigger_position"]
    )
    gripper_command = _validated_number(
        "gripper_command", decoded["gripper_command"]
    )
    return OmyStateMessage(
        protocol_version=version,
        sequence=sequence,
        source_timestamp_ns=source_timestamp_ns,
        sender_monotonic_ns=sender_monotonic_ns,
        joint_names=OMY_JOINT_NAMES,
        position=ordered,
        trigger_position=trigger_position,
        gripper_command=gripper_command,
    )
