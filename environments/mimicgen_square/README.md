# MimicGen Square FR3 Port

## Upstream Source

This is a `Square_D0` task port from NVIDIA MimicGen `1.0.1` commit
`72bd767c255545f462e7ccfb2731f2e5d4c1d9bb`. Its task base is robosuite
`NutAssemblySquare` at MimicGen's documented robosuite commit `b9d8d3de...`.
See [UPSTREAM_SQUARE_AUDIT.md](UPSTREAM_SQUARE_AUDIT.md) for exact sources and
values.

## What Was Ported

The square nut's five-box geometry and object physics, fixed square peg,
Square_D0 XY/yaw reset distribution, and source success predicate were ported.
The upstream task frame is rotated `+90 degrees` around world Z and translated
by `(0.595, 0, -0.395)` m. This maps its table top origin `(0, 0, 0.82)` to
the current FR3 table top `(0.595, 0, 0.425)` while retaining all relative
task geometry and placing the transfer axis inside the FR3 workspace.

## What Was Not Ported

The upstream robot, gripper, OSC/action API, control loop, cameras, MimicGen
demonstration generator, dataset writer, and policy stack were not ported.
This is not an exact benchmark reproduction.

The shared wrist camera semantics are retained with a Square-only local pose:
position `(0.07, -0.10, 0.0)` m, `30.96`-degree pitch, and the existing
`90`-degree vertical FOV. This documented visibility adaptation keeps the nut
and square peg in view around grasp, approach, and insertion. It is applied
only when this environment is constructed and does not alter Pick-and-Place.

## Current Robot Stack

The shared Franka FR3 model, current actuator/controller implementation, and
OMY-L100 Position (`1`), Rotation (`2`), clutch, and gripper paths remain
authoritative. Physics/control stays at 1000 Hz and recording stays at 20 Hz.

## Observation

Current policy observation is `operator`, `wrist`, and 16D robot state: seven FR3
joint positions, seven velocities, and two finger positions. Object and peg
poses are diagnostics only and are not policy inputs.

## Action

Action remains eight-dimensional: seven absolute FR3 joint targets plus the
native gripper command.

## Reset

`reset(seed=...)` restores canonical FR3 home and samples D0 nut X/Y/yaw with
a deterministic NumPy random state. The peg and cameras are fixed.

## Success

The nut must be within 30 mm of the peg on both X and Y, below table-top plus
50 mm, and more than approximately 42.365 mm from the released gripper.

## Data Path

Raw and processed data belong under `experiments/mimicgen_square/datasets/`;
training and evaluation outputs belong in the sibling experiment folders.

## Current Validation

Only checks actually run on this branch are listed here and updated with the
implementation tests: model loading, deterministic reset, 100-reset validity,
success-state checks, action/state dimensions, camera rendering, numerical
reachability, process-isolated camera rendering, initial-state restoration,
generic raw-to-LeRobot synthetic contract conversion, and Pick-and-Place
regression. Physical OMY teleoperation and a human-completed episode have not
been run automatically.
