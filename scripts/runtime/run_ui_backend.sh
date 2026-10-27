#!/usr/bin/env bash
set -Eeo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIMIC_PYTHON="${MIMIC_PYTHON:-$(command -v python3)}"
if [[ "${MIMIC_UI_LEADER:-omy}" != "geomagic" ]]; then
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    # Preserve ROS Python paths for the system-Python IPC child.
    source /opt/ros/humble/setup.bash
  fi
  omy_ros_ws="${OMY_ROS_WS:-$project_root/../franka-teleoperation/open_manipulator_omy}"
  omy_setup="$omy_ros_ws/install_final/setup.bash"
  if [[ ! -f "$omy_setup" ]]; then
    omy_setup="$omy_ros_ws/install/setup.bash"
  fi
  if [[ -f "$omy_setup" ]]; then
    source "$omy_setup"
  fi
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mimic-matplotlib-cache}"
export PYTHONPATH="$project_root/environments/mimic_robot:$project_root:${FRANKA_TELEOP_ROOT:-$project_root/../franka-teleoperation}${PYTHONPATH:+:$PYTHONPATH}"

cd "$project_root"
exec "$MIMIC_PYTHON" -m ui.experiment_console.backend.server "$@"
