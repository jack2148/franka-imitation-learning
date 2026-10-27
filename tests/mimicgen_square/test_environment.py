from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from environments.mimicgen_square.config import (
    FR3_MAX_FINGER_OPENING,
    NUT_HANDLE_FULL_WIDTH,
    PEG_POSITION,
    SUCCESS_MAX_NUT_Z,
)
from environments.mimicgen_square.task import SCENE_XML
from environments.mimicgen_square.validation import make_environment, validate_resets
from runtime.live_runner import MimicLiveRunner
from runtime.recording import AsyncCameraWriter


class MimicGenSquareEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_environment(image_size=(96, 128))

    def tearDown(self) -> None:
        self.env.close()

    def test_model_assets_and_policy_contract(self) -> None:
        self.assertTrue(SCENE_XML.is_file())
        self.assertTrue((SCENE_XML.parent / "assets" / "square_nut.xml").is_file())
        self.assertEqual(self.env.action_dim, 8)
        self.assertEqual(self.env.policy_state().shape, (16,))
        self.assertGreater(FR3_MAX_FINGER_OPENING, NUT_HANDLE_FULL_WIDTH)

    def test_reset_is_deterministic_by_seed(self) -> None:
        self.env.reset_state(seed=23)
        first = self.env.object_pose()
        self.env.reset_state(seed=23)
        second = self.env.object_pose()
        np.testing.assert_array_equal(first, second)
        self.env.reset_state(seed=24)
        self.assertFalse(np.array_equal(first, self.env.object_pose()))

    def test_one_hundred_resets_are_valid(self) -> None:
        result = validate_resets(self.env, 100)
        self.assertEqual(result["seeds_tested"], 100)
        self.assertEqual(result["invalid_resets"], 0, result["invalid_details"])
        np.testing.assert_allclose(result["target_position_m"], PEG_POSITION)

    def test_success_failure_and_incomplete_states(self) -> None:
        self.env.reset_state(seed=0)
        self.assertFalse(self.env.success())

        address = self.env.object_qpos_address
        self.env.data.qpos[address : address + 3] = [
            PEG_POSITION[0],
            PEG_POSITION[1],
            SUCCESS_MAX_NUT_Z + 0.01,
        ]
        self.env.data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertFalse(self.env.success(), "aligned but too-high nut is incomplete")

        self.env.data.qpos[address + 2] = SUCCESS_MAX_NUT_Z - 0.04
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertTrue(self.env.success())

    def test_operator_and_wrist_policy_observation(self) -> None:
        self.env.reset_state(seed=0)
        observation = self.env.policy_observation(render_images=True)
        self.assertEqual(observation["observation.state"].shape, (16,))
        self.assertEqual(observation["observation.images.operator"].shape, (96, 128, 3))
        self.assertEqual(observation["observation.images.wrist"].shape, (96, 128, 3))
        self.assertNotIn("observation.images.front", observation)
        self.assertTrue(np.isfinite(observation["observation.state"]).all())

    def test_action_validation_remains_eight_dimensional(self) -> None:
        self.env.reset_state(seed=0)
        home = self.env.model.key_qpos[self.env.model.key("home").id, :7]
        action = np.concatenate((home, [255.0]))
        self.env.apply_action(action)
        with self.assertRaises(ValueError):
            self.env.apply_action(np.zeros(7))


class SharedRuntimeRegressionTest(unittest.TestCase):
    def test_live_runner_supports_square_without_changing_pick_place_default(self) -> None:
        with MimicLiveRunner(input_backend="synthetic", image_size=(16, 16)) as pick:
            action, observation = pick.step_once(capture_observation=True)
            self.assertEqual(pick.task, "fr3-pick-place")
            self.assertEqual(action.shape, (8,))
            self.assertEqual(observation.observation_state.shape, (16,))
            self.assertAlmostEqual(float(pick.model.body("table").pos[0]), 0.595)
            np.testing.assert_allclose(
                pick.model.cam_pos[pick.model.camera("wrist").id],
                [0.0, -0.16, -0.04],
            )

        with MimicLiveRunner(
            task="mimicgen_square", input_backend="synthetic", image_size=(16, 16)
        ) as square:
            initial_qpos = square.initial_state.qpos.copy()
            action, observation = square.step_once(capture_observation=True)
            self.assertEqual(square.task, "mimicgen_square")
            self.assertEqual(action.shape, (8,))
            self.assertEqual(observation.observation_state.shape, (16,))
            self.assertFalse(square.robot_table_contact())
            np.testing.assert_allclose(
                square.model.cam_pos[square.model.camera("wrist").id],
                [0.07, -0.10, 0.0],
            )
            square.data.qpos[:] += 0.01
            square.reset_to_initial()
            np.testing.assert_array_equal(square.data.qpos, initial_qpos)
            np.testing.assert_array_equal(square.env.target_pos, PEG_POSITION)

    def test_square_camera_override_reaches_dataset_render_process(self) -> None:
        with MimicLiveRunner(
            task="mimicgen_square", input_backend="synthetic", image_size=(16, 16)
        ) as square, tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            writer = AsyncCameraWriter(
                square.xml_path,
                goal_position=square.initial_goal,
                target_body_name=square.target_body_name,
                camera_overrides=square.camera_overrides,
                episode_dir=output,
                image_size=(96, 128),
            )
            writer.submit(0, square.initial_state)
            writer.close()
            self.assertEqual(writer.written, 1)
            self.assertTrue((output / "images" / "operator" / "000000.png").is_file())
            self.assertTrue((output / "images" / "wrist" / "000000.png").is_file())
            self.assertFalse((output / "images" / "front").exists())
            for ui_only in ("front_operator", "left_side", "top_down"):
                self.assertFalse((output / "images" / ui_only).exists())


if __name__ == "__main__":
    unittest.main()
