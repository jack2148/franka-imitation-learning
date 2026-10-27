# Public v1 boundary

The release contains original OMY teleoperation and **ordinary Geomagic Touch** control, plus the paired Square recording / validation / conversion, baseline ACT / Diffusion Policy wrappers, evaluation modules and published result analysis.

Excluded extension families: the unreleased alternative Touch controller, Omega7, Mug Cleanup, Gear Assembly, weighting / LIW / OS training, and their dedicated adapters, launchers, tests and UI panels. These are scope exclusions, not passing or skipped tests. Original research workspace files and earlier publication snapshots remain intact. No ownership investigation was expanded to excluded code.

The base Pick-and-Place environment, its model assets and regression tests remain as dependencies of the existing shared runner; it is not an added headline v1 result. Manual folder isolation and operator 1/2/3 annotation remain required core collection functionality; storing labels does not enable a weighting algorithm.

Mixed tests were split by dependency: core label, pre-step physics and PNG byte checks remain; excluded-task recorder tests do not enter the v1 test tree. The SDK compile test is an explicitly separate dependency gate, not silently skipped. Hardware-free tests retain existing synthetic fixtures and do not replace device behavior with a new mock.

Control gains, coordinate mappings, DLS/null-space algorithms, clutch, success predicate, observation/action contract and saved training configs remain frozen. Publication path repairs and excluded-branch removal are separately recorded; later source changes do not auto-sync.
