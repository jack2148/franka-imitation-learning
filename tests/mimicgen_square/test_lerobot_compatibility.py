from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

os.environ.setdefault("HF_HOME", "/tmp/mimicgen_square_huggingface_test")
os.environ.setdefault(
    "HF_DATASETS_CACHE", "/tmp/mimicgen_square_huggingface_test/datasets"
)

from dataset.convert_mimic_raw_to_lerobot import convert
from dataset.validate_lerobot_dataset import validate_lerobot_dataset
from runtime.dataset_validation import validate_dataset


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "installed LeRobot is required")
class MimicGenSquareLeRobotCompatibilityTest(unittest.TestCase):
    def test_generic_converter_accepts_square_task_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "raw" / "synthetic_square_contract_check"
            episode.mkdir(parents=True)
            count = 2
            np.savez(
                episode / "samples_20hz.npz",
                sim_time=np.arange(count) * 0.05,
                control_step=np.arange(count, dtype=np.int64) * 50,
                dataset_frame_index=np.arange(count, dtype=np.int64),
                observation_state=np.zeros((count, 16)),
                arm_qpos=np.zeros((count, 7)),
                arm_qvel=np.zeros((count, 7)),
                gripper_qpos=np.zeros((count, 2)),
                action=np.zeros((count, 8)),
                ee_pose=np.zeros((count, 7)),
                object_pose=np.zeros((count, 7)),
                goal_position=np.zeros((count, 3)),
                success=np.ones(count, dtype=bool),
            )
            np.savez(episode / "initial_state.npz", qpos=np.zeros(16))
            (episode / "metadata.json").write_text(
                json.dumps(
                    {
                        "task": "mimicgen_square",
                        "experiment": "mimicgen_square",
                        "environment": "mimicgen_square",
                        "camera_schema": "operator_wrist_v1",
                        "outcome": "success",
                        "temporal_convention": "observation_t_before_step -> action_t",
                    }
                ),
                encoding="utf-8",
            )
            for camera in ("operator", "wrist"):
                image_dir = episode / "images" / camera
                image_dir.mkdir(parents=True)
                for frame_index in range(count):
                    Image.new("RGB", (640, 480), (20, 40, 60)).save(
                        image_dir / f"{frame_index:06d}.png"
                    )

            raw_validation = validate_dataset(episode)
            self.assertTrue(raw_validation["valid"], raw_validation)
            output = root / "processed" / "square_contract_check"
            provenance = convert(
                episode,
                output,
                task="mimicgen_square",
            )
            processed_validation = validate_lerobot_dataset(output, episode)
            self.assertEqual(provenance["task"], "mimicgen_square")
            self.assertEqual(provenance["frames"], count)
            self.assertEqual(provenance["camera_schema"], "operator_wrist_v1")
            self.assertTrue(processed_validation["valid"], processed_validation)
            self.assertEqual(processed_validation["action_shape"], (8,))
            self.assertEqual(processed_validation["state_shape"], (16,))
            self.assertTrue(processed_validation["operator_equivalent"])
            self.assertTrue(processed_validation["wrist_equivalent"])


if __name__ == "__main__":
    unittest.main()
