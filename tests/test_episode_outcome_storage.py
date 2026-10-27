"""Outcome-grouped raw episode storage: naming, sequencing, and discovery.

Covers the pure NNN_HHMM naming/sequence module, the validator's outcome-
folder cross-check, converter discovery of both layouts, and one full
end-to-end recording pass through the real WORK worker to confirm the
dataset contract (operator+wrist, 20 Hz, 16D/8D) is unchanged for a
grouped save.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import time
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/tmp/mimic_outcome_storage_huggingface_test")
os.environ.setdefault(
    "HF_DATASETS_CACHE", "/tmp/mimic_outcome_storage_huggingface_test/datasets"
)

import numpy as np
from PIL import Image

from dataset.convert_mimic_raw_to_lerobot import convert, discover_episodes
from runtime.dataset_validation import validate_dataset
from runtime.episode_naming import (
    finalize_grouped_episode,
    format_episode_name,
    is_grouped_episode_name,
    next_attempt_sequence,
)
from runtime.ui_worker import work_worker_main


def _make_scratch(parent: Path, tag: str) -> Path:
    scratch = parent / f"scratch_{tag}"
    scratch.mkdir()
    (scratch / "placeholder.txt").write_text("x", encoding="utf-8")
    return scratch


def _write_minimal_episode(path: Path, *, outcome: str, task: str = "pick_and_place") -> None:
    """Write the smallest episode validate_dataset() accepts as valid."""
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "task": task,
                "outcome": outcome,
                "temporal_convention": "observation_t_before_step -> action_t",
                "camera_schema": "operator_wrist_v1",
            }
        ),
        encoding="utf-8",
    )
    count = 2
    np.savez(
        path / "samples_20hz.npz",
        sim_time=np.array([0.0, 0.05]),
        control_step=np.array([0, 50]),
        dataset_frame_index=np.arange(count),
        observation_state=np.zeros((count, 16)),
        arm_qpos=np.zeros((count, 7)),
        arm_qvel=np.zeros((count, 7)),
        gripper_qpos=np.zeros((count, 2)),
        action=np.zeros((count, 8)),
        ee_pose=np.zeros((count, 7)),
        object_pose=np.zeros((count, 7)),
        goal_position=np.zeros((count, 3)),
        success=np.zeros(count, dtype=bool),
    )
    np.savez(path / "initial_state.npz", qpos=np.zeros(1))
    for camera in ("operator", "wrist"):
        image_dir = path / "images" / camera
        image_dir.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (640, 480)).save(image_dir / f"{index:06d}.png")


class EpisodeNamingUnitTest(unittest.TestCase):
    def test_format_and_discovery_pattern(self) -> None:
        self.assertEqual(format_episode_name(1, "0907"), "001_0907")
        self.assertEqual(format_episode_name(1000, "2358"), "1000_2358")
        self.assertTrue(is_grouped_episode_name("001_0907"))
        self.assertTrue(is_grouped_episode_name("1000_2358"))
        self.assertFalse(is_grouped_episode_name("pick_and_place_20260910_165746_success"))
        with self.assertRaises(ValueError):
            format_episode_name(0, "0907")
        with self.assertRaises(ValueError):
            format_episode_name(1, "907")

    def test_sequence_discovery_scans_both_outcome_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            (raw_root / "success" / "001_0900").mkdir(parents=True)
            (raw_root / "failure" / "004_0905").mkdir(parents=True)
            (raw_root / "success" / "002_0901").mkdir(parents=True)
            self.assertEqual(next_attempt_sequence(raw_root), 5)

    def test_sequence_ignores_legacy_and_non_matching_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            (
                raw_root
                / "success"
                / "omy_fr3_mimic_20hz_diagnostic_v1_20260910_165746_success"
            ).mkdir(parents=True)
            (raw_root / "success" / "001_0900").mkdir(parents=True)
            self.assertEqual(next_attempt_sequence(raw_root), 2)

    def test_no_existing_episodes_starts_at_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(next_attempt_sequence(Path(temporary) / "raw"), 1)

    def test_same_minute_recordings_get_unique_sequential_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            hhmm = "1432"
            first_target, first_sequence = finalize_grouped_episode(
                _make_scratch(Path(temporary), "a"),
                raw_root,
                "success",
                {"outcome": "success"},
                hhmm=hhmm,
            )
            second_target, second_sequence = finalize_grouped_episode(
                _make_scratch(Path(temporary), "b"),
                raw_root,
                "success",
                {"outcome": "success"},
                hhmm=hhmm,
            )
            self.assertNotEqual(first_target, second_target)
            self.assertEqual((first_sequence, second_sequence), (1, 2))
            self.assertEqual(first_target.name, "001_1432")
            self.assertEqual(second_target.name, "002_1432")
            # metadata attempt_index matches the folder sequence number.
            self.assertEqual(
                json.loads((first_target / "metadata.json").read_text())["attempt_index"], 1
            )
            self.assertEqual(
                json.loads((second_target / "metadata.json").read_text())["attempt_index"], 2
            )

    def test_finalize_never_overwrites_an_existing_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            target, _sequence = finalize_grouped_episode(
                _make_scratch(Path(temporary), "a"),
                raw_root,
                "success",
                {"outcome": "success"},
                hhmm="0907",
            )
            original_contents = sorted(path.name for path in target.iterdir())
            second_target, second_sequence = finalize_grouped_episode(
                _make_scratch(Path(temporary), "b"),
                raw_root,
                "success",
                {"outcome": "success"},
                hhmm="0907",
            )
            self.assertNotEqual(target, second_target)
            self.assertEqual(second_sequence, 2)
            self.assertEqual(sorted(path.name for path in target.iterdir()), original_contents)

    def test_invalid_outcome_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                finalize_grouped_episode(
                    _make_scratch(Path(temporary), "a"),
                    Path(temporary) / "raw",
                    "diagnostic",
                    {"outcome": "diagnostic"},
                )


class OutcomeFolderValidationTest(unittest.TestCase):
    def test_success_folder_with_success_metadata_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            _write_minimal_episode(raw_root / "success" / "001_0910", outcome="success")
            result = validate_dataset(raw_root)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["episodes"], 1)

    def test_failure_folder_with_failure_metadata_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            _write_minimal_episode(raw_root / "failure" / "002_0915", outcome="failure")
            result = validate_dataset(raw_root)
            self.assertTrue(result["valid"], result["errors"])

    def test_outcome_mismatch_between_folder_and_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            # Folder says success, metadata says failure -- must not silently pass.
            _write_minimal_episode(raw_root / "success" / "003_0920", outcome="failure")
            result = validate_dataset(raw_root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("does not match" in error for error in result["errors"]))

    def test_legacy_flat_episode_is_preserved_and_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            _write_minimal_episode(
                raw_root / "omy_fr3_mimic_20hz_diagnostic_v1_20260910_165746_success",
                outcome="success",
            )
            result = validate_dataset(raw_root)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["episodes"], 1)

    def test_mixed_legacy_and_grouped_layouts_both_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            _write_minimal_episode(
                raw_root / "omy_fr3_mimic_20hz_diagnostic_v1_20260910_165746_success",
                outcome="success",
            )
            _write_minimal_episode(raw_root / "success" / "001_0910", outcome="success")
            result = validate_dataset(raw_root)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["episodes"], 2)


class ConverterOutcomeDiscoveryTest(unittest.TestCase):
    def test_discover_episodes_finds_both_layouts_but_not_deeper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            _write_minimal_episode(raw_root / "success" / "001_0900", outcome="success")
            _write_minimal_episode(raw_root / "failure" / "002_0905", outcome="failure")
            _write_minimal_episode(raw_root / "legacy_flat_episode", outcome="success")
            found = {path.name for path in discover_episodes(raw_root)}
            self.assertEqual(found, {"001_0900", "002_0905", "legacy_flat_episode"})

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "installed LeRobot is required")
    def test_convert_default_includes_only_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            _write_minimal_episode(raw_root / "success" / "001_0900", outcome="success")
            _write_minimal_episode(raw_root / "failure" / "002_0905", outcome="failure")
            provenance = convert(raw_root, Path(temporary) / "processed", task="pick_and_place")
            self.assertEqual(provenance["episodes"], 1)
            self.assertEqual(
                [record["outcome"] for record in provenance["source_episodes"]], ["success"]
            )
            self.assertEqual(
                [item["episode"] for item in provenance["excluded_episodes"]], ["002_0905"]
            )
            self.assertEqual(
                [item["reason"] for item in provenance["excluded_episodes"]],
                ["outcome_not_selected"],
            )

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "installed LeRobot is required")
    def test_outcome_mismatch_blocks_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            # Placed under success/ but metadata says failure.
            _write_minimal_episode(raw_root / "success" / "001_0900", outcome="failure")
            with self.assertRaisesRegex(ValueError, "does not match"):
                convert(raw_root, Path(temporary) / "processed", task="pick_and_place")


class GroupedRecordingEndToEndTest(unittest.TestCase):
    """One real recording per outcome through the actual WORK worker."""

    def _run_episode(self, raw_root: Path, outcome: str, *, seconds: float = 1.0) -> dict:
        context = mp.get_context("spawn")
        commands = context.Queue()
        states = context.Queue(maxsize=1)
        process = context.Process(
            target=work_worker_main,
            args=(
                {
                    "region": "A",
                    "task": "pick_and_place",
                    "input_backend": "synthetic",
                    "seed": 0,
                    "dataset_version": "outcome_storage_test",
                    "dataset_output_root": str(raw_root),
                    "_preview_dir": str(raw_root.parent / "preview"),
                },
                commands,
                states,
            ),
        )
        process.start()
        deadline = time.monotonic() + 90.0
        stop_sent = False
        final_state: dict = {}
        try:
            while time.monotonic() < deadline:
                try:
                    state = states.get(timeout=0.2)
                    final_state = state
                except queue.Empty:
                    state = final_state
                if state.get("phase") == "ERROR":
                    raise RuntimeError(state.get("message"))
                if (
                    not stop_sent
                    and state.get("recording")
                    and float(state.get("elapsed_s") or 0) >= seconds
                ):
                    commands.put(outcome)
                    stop_sent = True
                if state.get("dataset_root"):
                    return state
                if not process.is_alive() and not state.get("dataset_root"):
                    raise RuntimeError(f"WORK worker exited early: {process.exitcode}")
            raise TimeoutError("WORK episode did not finish")
        finally:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    def test_success_then_failure_then_success_are_numbered_and_grouped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            first = self._run_episode(raw_root, "success")
            second = self._run_episode(raw_root, "failure")
            third = self._run_episode(raw_root, "success")

            first_path = Path(first["dataset_root"])
            second_path = Path(second["dataset_root"])
            third_path = Path(third["dataset_root"])

            self.assertEqual(first_path.parent.name, "success")
            self.assertEqual(second_path.parent.name, "failure")
            self.assertEqual(third_path.parent.name, "success")
            self.assertTrue(first_path.name.startswith("001_"))
            self.assertTrue(second_path.name.startswith("002_"))
            self.assertTrue(third_path.name.startswith("003_"))
            for path in (first_path, second_path, third_path):
                self.assertRegex(path.name, r"^\d{3,}_\d{4}$")

            for path, expected_sequence in (
                (first_path, 1),
                (second_path, 2),
                (third_path, 3),
            ):
                metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["attempt_index"], expected_sequence)
                self.assertIn("recorded_at", metadata)

            # Dataset contract unchanged: operator+wrist only, 20 Hz, 16D/8D.
            for path in (first_path, second_path, third_path):
                validation = validate_dataset(path)
                self.assertTrue(validation["valid"], validation["errors"])
                self.assertEqual(validation["camera_schema"], "operator_wrist_v1")
                self.assertEqual(validation["images"]["operator"], validation["samples"])
                self.assertEqual(validation["images"]["wrist"], validation["samples"])
                self.assertFalse((path / "images" / "front").exists())
                for ui_only in ("front_operator", "left_side", "top_down"):
                    self.assertFalse((path / "images" / ui_only).exists())
                self.assertTrue(validation["checks"]["timestamps"])
                self.assertTrue(validation["checks"]["action_shape"])
                with np.load(path / "samples_20hz.npz") as data:
                    self.assertEqual(data["observation_state"].shape[1], 16)
                    self.assertEqual(data["action"].shape[1], 8)

            self.assertEqual(next_attempt_sequence(raw_root), 4)
            aggregate = validate_dataset(raw_root)
            self.assertTrue(aggregate["valid"], aggregate["errors"])
            self.assertEqual(aggregate["episodes"], 3)


if __name__ == "__main__":
    unittest.main()
