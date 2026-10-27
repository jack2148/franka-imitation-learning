# Installation and validation

Use the paired repository directories side by side: `franka-teleoperation/` and `franka-imitation-learning/`. The original OMY layout stays at the teleoperation root; no shared controller extraction was performed. Square / Touch integration explicitly depends on both checkouts. No original research workspace is required on `PYTHONPATH`.

## Python / native dependencies

Validated on Linux with Python3.12.14 and MuJoCo3.12.0. A display is needed for interactive viewing; CPU tests use OSMesa (`libosmesa6` on Ubuntu). Native force tests need a C++17 compiler. Device use separately requires OpenHaptics headers/runtime, correct device permissions and the physical Touch. The SDK is not redistributed.

From the learning checkout, in a dedicated environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip install 'lerobot @ git+https://github.com/huggingface/lerobot.git@3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e'
. scripts/env.sh
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa python -m pytest tests ../franka-teleoperation/tests ../franka-teleoperation/geomagic/tests
python -m scripts.training.train_act --help
python -m scripts.training.train_diffusion_policy --help
python -m scripts.evaluation.evaluate_policy --help
python -m scripts.analysis.regenerate_test50 --output /tmp/square-public-report
```

CPU tests do not initialize a device or start training. Existing synthetic protocol / owner fixtures remain test-only. The SDK-dependent compile test is separated under the teleoperation `hardware_validation/` directory and must be requested explicitly after installing the SDK. ROS/device launchers are not part of the hardware-free pytest collection.

The local clean-tree check uses a relocated repository pair and an isolated copy of the pinned LeRobot package with already installed Python dependencies. It is not a fresh-OS, fresh-driver or physical-device installation certificate. The analysis script was additionally checked with the host plotting environment. A full dependency resolution on a new OS is not claimed.

## Ordinary Touch UI

Node22 or newer, npm and a browser are required (local build checked with Node26.8.1). From the learning checkout:

```sh
(cd ui/experiment_console/frontend && npm ci && npm run build)
. scripts/env.sh
scripts/ui/ui_geo_sq --print-defaults
# Device-dependent: only when the device is connected and its SDK is installed:
scripts/ui/ui_geo_sq
```

The ordinary collector has manual folder selection, success/failure/execution counters, explicit recording and numeric 1/2/3 labels. `--canonical` selects the legacy plan path and retains its integrity checks; it is not historical source reconstruction. Preview cameras are not automatically additional policy input cameras.

Actual collection, training and reevaluation require user-provided paths and external data/model payloads. Never treat synthetic tests or CSV regeneration as a policy success evaluation.

## OMY gate still blocked

The existing OMY `teleop`, `simul` and `launch/` paths are preserved. They require ROS2 Humble, `rclpy`, `sensor_msgs`, `std_msgs`, the OMY bringup workspace and optionally `franka_msgs`. ROS is an external dependency and was not started.

Additionally, the frozen public snapshot lacks `mujoco_menagerie/franka_fr3/scene.xml` (including the custom front wall) and `robotis_mujoco_menagerie/robotis_omy/scene.xml` at the OMY repository root. An unmodified upstream download or the Square model is not assumed identical. No substitute scene was installed. Recover and review these exact scene assets/provenance before calling OMY's clean-clone executable gate complete.
