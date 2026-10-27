# Upstream Square audit

## Upstream Identification

- Project/package: NVIDIA MimicGen `1.0.1`, commit
  `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb`.
- MimicGen environment: `Square_D0` in
  `mimicgen/envs/robosuite/nut_assembly.py`.
- Actual base environment: robosuite `NutAssemblySquare` (single-object
  `NutAssembly` with `nut_type="square"`).
- Audited robosuite source: MimicGen's documented compatibility commit
  `b9d8d3de5e3dfd1724f4a0e6555246c460407daa` (v1.4.1 lineage), especially
  `robosuite/environments/manipulation/nut_assembly.py`.
- Object source: `robosuite/models/objects/xml_objects.py` and
  `robosuite/models/assets/objects/square-nut.xml`.
- Fixture source: `robosuite/models/arenas/pegs_arena.py` and
  `robosuite/models/assets/arenas/pegs_arena.xml`.
- No local MimicGen or robosuite installation was present. The sources above
  were inspected from temporary official Git checkouts; neither repository was
  vendored wholesale.

## Object

- Name/class: `SquareNut` / `SquareNutObject`.
- Pose: free joint; reset quaternion is MuJoCo `wxyz` for a sampled world-Z
  yaw.
- Collision: five boxes. Ring side half-sizes are
  `(0.0105, 0.04375, 0.01)` and `(0.03125, 0.0105, 0.01)`; handle half-size is
  `(0.02525, 0.015875, 0.01)` at local `(0.054, 0, 0)`.
- Physical footprint: local X `[-0.04375, 0.07925]` m and local Y
  `[-0.04375, 0.04375]` m before yaw. Source placement radius is `0.11` m.
- Density: `100 kg/m^3`; mass derived from the five collision boxes is
  `0.01580675 kg` (visual-only duplicates excluded).
- Contact: `condim=4`, `solimp="0.998 0.998 0.001"`,
  `solref="0.02 1"`; friction is `(0.95, 0.3, 0.2)` on one side and
  `(0.95, 0.3, 0.1)` on the remaining boxes.
- XML placement metadata gives bottom offset `-0.05` m and top offset
  `+0.01` m.

## Target / Peg

- D0 uses fixed `peg1`; the round peg corresponds to the removed round nut.
- Upstream pose: `(0.23, 0.10, 0.85)` m in the arena world frame.
- Collision geometry: box half-size `(0.016, 0.016, 0.10)` m.
- Friction: `(1, 0.005, 0.0001)`.

## Reset Distribution

- Square nut X: uniform `[-0.115, -0.110]` m relative to table origin.
- Square nut Y: uniform `[0.110, 0.225]` m relative to table origin.
- Square nut yaw: uniform `[0, 2*pi]` around world Z.
- Sampler reference: `(0, 0, 0.82)` m, Z offset `+0.02` m, with source
  bottom offset `-0.05` m. The resulting root Z is `0.89` m before settling.
- `ensure_object_boundary_in_range=False` and
  `ensure_valid_placement=True`.
- Peg pose is fixed in D0. D1 and D2 randomize the peg and broaden the nut
  distribution; those variants are not this port.
- `NutAssemblySquare` also creates a round nut, samples it around Y
  `[-1.1, -1.0]`, then clears it because only the square nut is active.
- Upstream robot initialization follows robosuite defaults and is not a task
  semantic retained by this FR3 port.

## Success Condition

robosuite first checks the square nut independently on each horizontal axis:
`abs(nut.x - peg.x) < 0.03` and `abs(nut.y - peg.y) < 0.03`. The nut root must
also be below `table_top + 0.05` m. Finally, the gripper must be released far
enough that `1 - tanh(10 * distance(gripper, nut)) < 0.6`, equivalent to an
EE-to-nut distance greater than approximately `0.042365` m. In single-square
mode that one placed nut completes the task. Orientation is not explicitly
tested by the success function; physical peg/hole collision geometry enforces
insertion feasibility.

## Upstream Robot / Controller Assumptions

The source environment constructs a robosuite single-arm robot, its default
gripper, controller/action convention, 20 Hz environment control, robosuite
camera observations, and robosuite initialization noise. None of those define
the Square task geometry or success semantics, and none are retained here.

## Porting Decisions

| Feature | Decision | Reason |
|---|---|---|
| Square nut geometry and local sites | KEEP | Task-defining geometry. |
| Nut density, friction, and contact parameters | KEEP | Task-object physics. |
| D0 nut XY/yaw distribution | KEEP | Defines the selected MimicGen difficulty. |
| Fixed square peg geometry and relative pose | KEEP | Defines the target fixture. |
| Success predicate | KEEP | Defines task completion. |
| Table-frame coordinates | ADAPT | One +90-degree Z rotation and translation maps the complete upstream task rigidly into the reachable FR3 table frame. |
| Table dimensions and global solver | ADAPT | Reuse the validated FR3 scene instead of replacing project physics. |
| FR3 home state | ADAPT | Canonical project reset replaces upstream robot initialization. |
| Hidden round nut and round peg | DROP | They are inactive implementation artifacts in single-square mode. |
| Panda / robosuite controller and action API | DROP | Shared FR3–OMY control remains authoritative. |
| MimicGen generation, recorder, cameras, and policy stack | DROP | Existing project infrastructure remains authoritative; only a documented Square-local shared-wrist pose/FOV override is used for task visibility. |

This implementation is a **MimicGen Square task port**, not an exact benchmark
reproduction.
