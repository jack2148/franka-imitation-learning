from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from runtime.camera_schema import CURRENT_CAMERA_SCHEMA, LEGACY_CAMERA_SCHEMA
from runtime.dataset_validation import validate_dataset
from runtime.live_runner import MimicLiveRunner, STEPS_PER_DATASET_SAMPLE
from runtime.recording import AsyncCameraWriter, AsyncLivePreview, stack_observations


class CanonicalCameraSchemaTest(unittest.TestCase):
    def test_synthetic_recording_saves_only_aligned_operator_and_wrist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, MimicLiveRunner(
            input_backend="synthetic"
        ) as runner:
            root = Path(temporary)
            observations = []
            writer = AsyncCameraWriter(
                runner.xml_path,
                goal_position=runner.initial_goal,
                target_body_name=runner.target_body_name,
                camera_overrides=runner.camera_overrides,
                episode_dir=root,
            )
            for step in range(2 * STEPS_PER_DATASET_SAMPLE + 1):
                _, observation = runner.step_once(
                    capture_observation=step % STEPS_PER_DATASET_SAMPLE == 0
                )
                if observation is not None:
                    observations.append(observation)
                    writer.submit(len(observations) - 1, observation.physics)
            writer.close()

            samples = stack_observations(observations)
            np.savez(root / "samples_20hz.npz", **samples)
            np.savez(root / "initial_state.npz", qpos=runner.initial_state.qpos)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "task": "pick_and_place",
                        "environment": "fr3-pick-place",
                        "outcome": "success",
                        "camera_schema": CURRENT_CAMERA_SCHEMA,
                        "camera_config_version": CURRENT_CAMERA_SCHEMA,
                        "temporal_convention": "observation_t_before_step -> action_t",
                    }
                ),
                encoding="utf-8",
            )

            result = validate_dataset(root)
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["camera_schema"], CURRENT_CAMERA_SCHEMA)
            self.assertEqual(result["samples"], 3)
            self.assertEqual(result["images"].get("operator"), 3)
            self.assertEqual(result["images"].get("wrist"), 3)
            self.assertTrue(result["checks"]["timestamps"])
            self.assertTrue(result["checks"]["control_alignment"])
            self.assertTrue(result["checks"]["action_shape"])
            self.assertEqual(samples["observation_state"].shape, (3, 16))
            for ui_only in ("front", "front_operator", "left_side", "top_down"):
                self.assertFalse((root / "images" / ui_only).exists())

    def test_ui_preview_renders_left_side_but_not_wrist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, MimicLiveRunner(
            input_backend="synthetic", image_size=(96, 128)
        ) as runner:
            root = Path(temporary)
            preview = AsyncLivePreview(
                runner.xml_path,
                goal_position=runner.initial_goal,
                target_body_name=runner.target_body_name,
                camera_overrides=runner.camera_overrides,
                live_dir=root,
                image_size=(96, 128),
            )
            preview.submit(runner.capture_physics())
            deadline = time.monotonic() + 20.0
            while preview.rendered < 1 and time.monotonic() < deadline:
                time.sleep(0.05)
            preview.close()
            self.assertEqual(preview.rendered, 1)
            self.assertTrue((root / "operator_camera.jpg").is_file())
            self.assertTrue((root / "front_operator_camera.jpg").is_file())
            self.assertTrue((root / "left_side_camera.jpg").is_file())
            self.assertFalse((root / "wrist_camera.jpg").exists())
            self.assertFalse((root / "top_down_camera.jpg").exists())

    def test_legacy_schema_is_identified_without_operator_reinterpretation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for camera in ("front", "wrist"):
                directory = root / "images" / camera
                directory.mkdir(parents=True)
                from PIL import Image

                Image.new("RGB", (640, 480)).save(directory / "000000.png")
            np.savez(
                root / "samples_20hz.npz",
                sim_time=np.array([0.0]),
                control_step=np.array([0]),
                dataset_frame_index=np.array([0]),
                observation_state=np.zeros((1, 16)),
                arm_qpos=np.zeros((1, 7)),
                arm_qvel=np.zeros((1, 7)),
                gripper_qpos=np.zeros((1, 2)),
                action=np.zeros((1, 8)),
                ee_pose=np.zeros((1, 7)),
                object_pose=np.zeros((1, 7)),
                goal_position=np.zeros((1, 3)),
                success=np.zeros(1, dtype=bool),
            )
            np.savez(root / "initial_state.npz", qpos=np.zeros(1))
            (root / "metadata.json").write_text(
                '{"task":"pick_and_place","outcome":"success"}', encoding="utf-8"
            )
            result = validate_dataset(root)
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["camera_schema"], LEGACY_CAMERA_SCHEMA)
            self.assertNotIn("operator", result["images"])


if __name__ == "__main__":
    unittest.main()
