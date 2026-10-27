"""Synthetic operator-label tests. No device, production collection or training."""
import csv
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from runtime.sample_weight import SampleWeight, validate_weight, validate_weight_csv, save_weighted_samples
from runtime.recording import stack_observations

class SampleWeightTest(unittest.TestCase):

    def setUp(self):
        base = Path(os.environ.get('MIMIC_WEIGHT_TEST_OUTPUT', Path(__file__).resolve().parents[2] / 'outputs/realtime_weight_tests'))
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='sample_weight_', dir=base)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_exact_sequence_rapid_changes_no_retroactive_labels(self):
        labels = SampleWeight()
        for index in range(80):
            if index in (20, 40, 60):
                labels.set({20: 2, 40: 3, 60: 1}[index], labels.recording_id, str(index))
            labels.append(index)
        np.testing.assert_array_equal(labels.arrays(80)['weight'], np.repeat([1, 2, 3, 1], 20))
        previous = labels.arrays(80)['weight'].copy()
        for value in (1, 3, 2):
            labels.set(value, labels.recording_id, str(value))
        labels.append(80)
        self.assertEqual(labels.values[-1], 2)
        np.testing.assert_array_equal(previous, labels.values[:-1])

    def test_reset_default_and_stale_episode_rejected(self):
        labels = SampleWeight()
        old = labels.recording_id
        labels.set(3, old, 'old')
        labels.append(0)
        labels.reset()
        self.assertEqual(labels.current_weight, 1)
        self.assertFalse(labels.set(2, old, 'late'))
        labels.append(0)
        self.assertEqual(labels.values, [1])
        self.assertNotEqual(labels.recording_id, old)

    def test_invalid_values_counts_and_legacy_are_explicit(self):
        labels = SampleWeight()
        for value in (0, 4, True, False, '2', 2.0, None, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                labels.set(value, labels.recording_id, 'id')
        with self.assertRaises(ValueError):
            labels.append(1)
        with self.assertRaises(ValueError):
            labels.arrays(1)
        self.assertIsNone(validate_weight({}, 2))
        with self.assertRaises(ValueError):
            validate_weight({}, 2, {'sample_weight': labels.metadata()})
        for values in ([0, 1], [1, 4], [1.0, 2.0], ['1', '2'], [[1], [2]], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_weight({'weight': np.array(values)}, 2)

    def test_csv_all_previous_numeric_fields_unchanged_weight_rightmost(self):
        labels = SampleWeight()
        from runtime.live_runner import MimicLiveRunner
        with MimicLiveRunner(input_backend='synthetic', task='mimicgen_square') as runner:
            observations = []
            for i in range(80 * 50):
                _, obs = runner.step_once(capture_observation=i % 50 == 0)
                if obs is not None:
                    observations.append(obs)
            samples = stack_observations(observations)
        original = {k: v.copy() for (k, v) in samples.items()}
        for i in range(80):
            labels.set([1, 2, 3, 1][i // 20], labels.recording_id, str(i))
            labels.append(i)
        save_weighted_samples(self.root, samples, labels)
        with np.load(self.root / 'samples_20hz.npz') as saved:
            validate_weight_csv(self.root, saved)
            for (key, value) in original.items():
                np.testing.assert_array_equal(saved[key], value)
        with (self.root / 'samples_20hz.csv').open() as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][-1], 'weight')
        self.assertEqual(rows[0].count('weight'), 1)
        self.assertEqual(len(rows), 81)
        self.assertEqual([int(r[-1]) for r in rows[1:]], labels.values)
        rows[3][-1] = '2.0'
        with (self.root / 'samples_20hz.csv').open('w', newline='') as f:
            csv.writer(f).writerows(rows)
        with self.assertRaises(ValueError):
            validate_weight_csv(self.root, samples)

    def test_backend_queues_only_and_rejects_stale_or_off_requests(self):
        from ui.experiment_console.backend.work_adapter.runtime import WorkRuntime
        manager = WorkRuntime.__new__(WorkRuntime)
        manager._lock = threading.RLock()
        manager._states = queue.Queue()
        manager._commands = queue.Queue()
        manager._process = SimpleNamespace(is_alive=lambda : True)
        manager._last_state = {'recording': True, 'weight': {'recording_id': 'episode', 'current_weight': 1}}
        manager._drain_states = lambda : None
        result = manager.set_weight(3, 'episode', 'id')
        self.assertTrue(result['queued'])
        self.assertEqual(manager._last_state['weight']['current_weight'], 1)
        self.assertEqual(manager._commands.get_nowait(), dict(command='weight', weight=3, recording_id='episode', request_id='id'))
        with self.assertRaises(ValueError):
            manager.set_weight(2, 'old', 'id')
        with self.assertRaises(ValueError):
            manager.set_weight(True, 'episode', 'id')
        manager._last_state['recording'] = False
        with self.assertRaises(ValueError):
            manager.set_weight(2, 'episode', 'id')

    def test_real_physics_callback_is_before_integration_with_latest_weight(self):
        from runtime.live_runner import MimicLiveRunner
        from geomagic.bridge.udp_state_client import GeomagicUDPStateClient

        def receiver(_address, **kwargs):
            return GeomagicUDPStateClient(('127.0.0.1', 0), **kwargs)
        for task in ('mimicgen_square',):
            with self.subTest(task=task), patch('geomagic.leader.teleop_backend.GeomagicUDPStateClient', side_effect=receiver):
                runner = MimicLiveRunner(input_backend='geomagic', task=task, seed=0)
                labels = SampleWeight()
                rows = []
                try:

                    def before(obs):
                        self.assertEqual(float(runner.data.time), obs.sim_time)
                        self.assertEqual(obs.physics.time, obs.sim_time)
                        np.testing.assert_array_equal(obs.observation_state, runner.env.policy_state())
                        np.testing.assert_array_equal(obs.action[:7], runner.data.ctrl[runner.backend.fr3_actuator_indices])
                        labels.append(len(rows))
                        rows.append(obs)
                    for step in range(151):
                        if step == 49:
                            for v in (1, 3, 2):
                                labels.set(v, labels.recording_id, str(v))
                        if step == 51:
                            labels.set(3, labels.recording_id, '3')
                        runner.step_once(capture_observation=step % 50 == 0, before_step=before if step % 50 == 0 else None)
                    self.assertEqual(labels.values, [1, 2, 3, 3])
                    np.testing.assert_allclose([r.sim_time for r in rows], np.arange(4) * 0.05, atol=1e-12)
                finally:
                    runner.close()
if __name__ == '__main__':
    unittest.main()
