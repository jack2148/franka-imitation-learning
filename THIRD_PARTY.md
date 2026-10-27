# Third-party and source provenance

- Vendored Mimic environment: commit `982d495ca77f3dbc53a8f8f267c6f623bdf8e5b8`, MIT notice in [upstream/LICENSE](environments/mimic_robot/upstream/LICENSE), source in [PROVENANCE](environments/mimic_robot/upstream/PROVENANCE.md).
- Square is a local FR3 port with changed robot/action/camera integration, not an exact MimicGen benchmark reproduction. [Upstream audit](environments/mimicgen_square/UPSTREAM_SQUARE_AUDIT.md) and [robosuite MIT notice](environments/mimicgen_square/assets/ROBOSUITE_LICENSE) record the task provenance.
- FR3/Panda/hand assets retain Apache2.0 notices in their asset folders. OMY assets used by the shared helper retain their included notices under `teleop/omy_fr3/robotis_mujoco_menagerie/`. These are not asserted identical to the missing native OMY launcher scene.
- LeRobot is an external dependency at commit `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`; its license remains upstream. Its models, trainer algorithms and dependency payloads are not copied into this repository.
- Frontend dependency versions and package integrity hashes remain in `package-lock.json`. Full installed packages / build output are excluded.
- OpenHaptics SDK and device binaries are external, not redistributed.

Project-authored files retain their existing notices. No new blanket license grant is invented by this migration. The author confirmed publication rights for the original Geomagic reference implementation; that confirmation does not extend to excluded controller extensions. Full dataset / checkpoint distribution is separate and has not occurred.
