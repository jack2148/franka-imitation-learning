"""Relocation and preserved contract checks; no device, model inference or training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def launcher_defaults(tmp_path, flags, extra):
    launcher = tmp_path / 'ui_geo_sq'
    launcher.symlink_to(ROOT / 'scripts/ui/run_ui.sh')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('MIMIC_', 'VITE_', 'GEOMAGIC_'))}
    env.update(extra)
    result = subprocess.run([str(launcher), *flags, '--print-defaults'],
                            env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('extra', [{}, {'MIMIC_GEOMAGIC_HAPTICS': '0'},
    {'MIMIC_GEOMAGIC_HAPTICS': '1', 'GEOMAGIC_SYNC_AUTO': '1'}])
def test_ordinary_manual_launcher_keeps_existing_no_capture_policy(tmp_path, extra):
    value = launcher_defaults(tmp_path, [], extra)
    assert value['collection_plan_id'] is None
    assert value['sync_capture_root'] is None
    assert value['dataset_version'] == 'geo_sq_weight_v1'
    assert value['geomagic_source'] == 'physical_hdapi'
    assert value['haptics_enabled'] is (extra.get('MIMIC_GEOMAGIC_HAPTICS', '1') == '1')
    # --print-defaults is a pure read: the collection workspace is not created.
    assert not Path(value['collection_workspace']).exists()


def test_historical_context_is_still_rejected_instead_of_resealed():
    from runtime.stratified_plan import load_manifest
    with pytest.raises(ValueError, match='plan/config/fixed-context mismatch'):
        load_manifest(directory=ROOT / 'evidence/collection')


def test_coordinate_reader_preserves_seal_and_explicit_non_equivalence():
    from runtime.touch_weight_collection import source_training_plan
    from runtime.stratified_plan import PLAN_DIRECTORY, TRAIN_NAME
    document, provenance = source_training_plan()
    assert document['content_sha256'] == '48bae8997f1aca79c5d8a6513f377d9c24589cf800b99e66497148e925e3318b'
    assert len(document['states']) == 50
    assert provenance['original_environment_equivalence_claimed'] is False
    assert provenance['canonical_plan_acceptance'] is False
    assert not (PLAN_DIRECTORY / (TRAIN_NAME + '.frozen.json')).exists()


def test_all_50_training_coordinate_resets_use_the_saved_order():
    from runtime.live_runner import MimicLiveRunner
    from runtime.touch_weight_collection import source_training_plan, apply_research_initial_state
    document, _ = source_training_plan()
    ordered = sorted(document['states'], key=lambda s: s['collection_order'])
    with MimicLiveRunner(input_backend='synthetic', task='mimicgen_square', image_size=(16, 16)) as runner:
        for i, state in enumerate(ordered):
            record = apply_research_initial_state(runner, i)
            address = runner.object_qpos_address
            np.testing.assert_array_equal(runner.data.qpos[address:address+3],
                [state['x'], state['y'], document['object_z']])
            assert record['selected_state'] == state
            assert record['coordinate_cycle'] == 0
        assert apply_research_initial_state(runner, 50)['selected_state'] == ordered[0]


def test_ordinary_adapter_selection_without_starting_hardware():
    # A new interpreter observes the real launcher environment at import time.
    # synthetic_udp is an existing supported source; no owner call is mocked.
    env = {k:v for k,v in os.environ.items() if not k.startswith(('MIMIC_', 'VITE_', 'GEOMAGIC_'))}
    env.update(MIMIC_UI_LEADER='geomagic', MIMIC_UI_TASK='mimicgen_square',
               MIMIC_UI_GEOMAGIC_SOURCE='synthetic_udp', MIMIC_WEIGHT_COLLECTION='1',
               MIMIC_COLLECTION_WORKSPACE=str(ROOT / 'outputs/research/adapter_contract'))
    script = '''from ui.experiment_console.backend.work_adapter import WorkAdapter
from ui.experiment_console.backend.work_adapter.touch_weight_runtime import TouchWeightRuntime
a=WorkAdapter()
try:
 assert isinstance(a.runtime, TouchWeightRuntime)
 assert a.runtime._process is None
 assert a.runtime._geomagic_device is None
 assert a.runtime._launch_profile.collection_plan_id is None
 assert a.runtime._collection_progress() is None
finally:a.close()
'''
    subprocess.run([sys.executable, '-B', '-c', script], env=env, check=True)


def test_storage_guard_still_rejects_frozen_outputs(tmp_path):
    from runtime.collection_counts import validate_output
    protected = ROOT / 'experiments/mimicgen_square/datasets/raw/restricted30_geomagic'
    protected.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match='fixed data is protected'):
        validate_output(protected)
    assert validate_output(tmp_path) == tmp_path.resolve()


def test_numeric_label_sequence_and_episode_reset_are_unchanged():
    from runtime.sample_weight import SampleWeight
    labels = SampleWeight()
    for index in range(80):
        if index in (20, 40, 60):
            labels.set({20:2,40:3,60:1}[index], labels.recording_id, str(index))
        labels.append(index)
    np.testing.assert_array_equal(labels.arrays(80)['weight'], np.repeat([1,2,3,1],20))
    labels.reset()
    assert labels.current_weight == 1
