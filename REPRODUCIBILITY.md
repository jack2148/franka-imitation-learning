# Reproducibility boundary

## Completed result identity

The completed result is ACT300k **27/50 (54%)** versus Diffusion Policy480k **43/50 (86%)** on the same frozen Square TEST50. The exported state JSON preserves all 50 state coordinates, order, yaw and inference seeds. It is a selected-field public derivative, not the original sealed file. [Export hashes and transformations](results/square_test50/EXPORT_MANIFEST.json) identify the original and public bytes.

[Dataset identity](results/square_test50/DATASET_IDENTITY.json) records the observed metadata SHA256, 50 episodes, 48,272 frames, 20 Hz and feature schema of `square_geomagic_xy50_v1_convert`. This is metadata identity, not a claimed full dataset content checksum. No demonstration bytes were changed or redistributed here.

[Terminal artifact identities](results/square_test50/ARTIFACT_MANIFEST.json) identify model and normalizer hashes. Model payloads remain external. Public config JSONs redact private storage paths; their export hashes identify the transformations. Small normalizer tensors retain original bytes.

## Fixed evaluation protocol

- 5×5 workspace cells, two states per cell; X [0.445, 0.485] m, Y [−0.1425, −0.0825] m; fixed nut yaw π/2.
- Saved shuffled order, per-state inference seed 20265000 + order, identical for the policies.
- MuJoCo physics 1000 Hz, policy 20 Hz, up to 2,400 actions / 120 simulated seconds.
- Operator + wrist RGB 640×480; state16 = q7/dq7/fingers2, action8 = absolute qtarget7/native gripper command (0 closed, 255 open).
- Success remains the implemented Square geometric predicate. Acquisition is a sustained lift/proximity diagnostic, not measured finger contact. The exported A–F classification does not change task termination.
- Checkpoint development curves and final TEST50 are distinct. Do not select a checkpoint or tune settings with TEST50.

The generic evaluation CLI imports and loads configurations, but does **not** implement the historical frozen-state campaign launcher. Its ordinary seed-reset mode must not be described as rerunning this frozen TEST50. Full policy reevaluation additionally requires the external checkpoints and a reviewed frozen-state orchestration path.

## Training configuration

The complete saved training/policy configurations are under `configs/square/act300k/` and `dp480k/`. ACT uses batch16, obs1, chunk50, action10, LR1e−5, seed0; terminal300k followed a resume from140k. DP uses batch16, obs8, horizon32, action8, LR1e−4, seed0, terminal480k and100 denoising steps. Inspect the saved configurations for all remaining settings.

`wrapper.json` is a mechanically derived subset of each saved config for the existing lightweight wrapper API. It changes no numeric values, but the wrapper subset is **not** a full historical training command or a guarantee that defaults in a later trainer are identical. UI overrides are explicit new-run settings. No historical config or research algorithm was patched to force current compatibility.

## What can be reproduced now

- Published per-state totals, spatial groups and failure-stage counts; representative PNG and CSV regeneration.
- CPU Square initialization / recording and conversion contracts with synthetic fixtures.
- Frozen public Touch control behavior against the previously validated snapshot.
- Current entry point imports, saved config decoding and command construction.

## What remains unavailable or unverified

- Byte-identical reconstruction of all historical TRAIN controller/training source. The available historical TRAIN plan and seal are retained under `evidence/collection`; their integrity gate is not re-sealed or bypassed.
- Full dataset, terminal model downloads, optimizer/RNG states and action-trajectory logs. Source hashes and small evidence do not replace these payloads.
- Historical stage reclassification from raw trajectories (the new script aggregates stored stages), exact retraining, policy success-rate replication, device haptic acceptance, physical FR3 performance and end-to-end real-time behavior.

Ordinary current Touch collection uses a separate manual output path and preserved coordinate reader. Using historical XY coordinates does not establish equivalence to the historical control environment.

## Source cutoff

[Source cutoff](provenance/SOURCE_CUTOFF.json) identifies the snapshot validated with655 tests and2,800 exact-comparison control steps on2026-10-27. Later workspace gain changes are not synchronized. V1 edits only narrow scope and repair relocation/config-path infrastructure; future controller work belongs to a later revision.
