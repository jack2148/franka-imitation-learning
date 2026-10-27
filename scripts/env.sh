#!/usr/bin/env bash
# Source from a paired checkout; no original research workspace is referenced.
learning_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export FRANKA_LEARNING_ROOT="$learning_root"
export FRANKA_TELEOP_ROOT="${FRANKA_TELEOP_ROOT:-$learning_root/../franka-teleoperation}"
export PYTHONPATH="$learning_root:$learning_root/environments/mimic_robot:$FRANKA_TELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MIMIC_PYTHON="$(command -v python)"
unset learning_root
