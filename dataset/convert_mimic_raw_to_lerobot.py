"""Convert aligned Mimic raw episodes with the official LeRobot v3 writer."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable
import numpy as np
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from runtime.dataset_validation import validate_dataset as validate_mimic_raw
from runtime.camera_schema import camera_names_for_schema
from runtime.episode_naming import OUTCOME_DIRS
from runtime.sample_weight import validate_weight
FPS = 20
STATE_NAMES = tuple([f'fr3_joint{i}_position' for i in range(1, 8)] + [f'fr3_joint{i}_velocity' for i in range(1, 8)] + ['fr3_finger1_position', 'fr3_finger2_position'])
ACTION_NAMES = tuple([f'fr3_joint{i}_target' for i in range(1, 8)] + ['fr3_gripper'])
BASE_FEATURES = {'observation.state': {'dtype': 'float32', 'shape': (16,), 'names': STATE_NAMES}, 'action': {'dtype': 'float32', 'shape': (8,), 'names': ACTION_NAMES}}

def features_for_schema(schema: str) -> dict:
    features = dict(BASE_FEATURES)
    for camera in camera_names_for_schema(schema):
        features[f'observation.images.{camera}'] = {'dtype': 'image', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channel']}
    return features

def _git_head(root: Path) -> str:
    result = subprocess.run(('git', '-C', str(root), 'rev-parse', 'HEAD'), capture_output=True, text=True, check=False)
    return result.stdout.strip() or 'unknown'

def _looks_like_episode(path: Path) -> bool:
    return path.is_dir() and ((path / 'metadata.json').exists() or (path / 'samples_20hz.npz').exists())

def discover_episodes(input_root: Path) -> list[Path]:
    """Find raw episodes under a raw root, both layouts side by side.

    Looks directly under ``input_root`` for legacy flat episodes and under
    ``input_root/success`` and ``input_root/failure`` for the current
    outcome-grouped layout -- nothing deeper. Which episodes actually make
    it into a converted dataset is still decided by the caller's outcome
    filter (``include_outcomes``, default success-only), read from each
    episode's own metadata, not from which folder it happened to be in.
    """
    root = input_root.expanduser().resolve()
    if (root / 'metadata.json').is_file() or (root / 'samples_20hz.npz').is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f'raw input root does not exist: {root}')
    candidates: list[Path] = [path for path in root.iterdir() if path.is_dir() and path.name not in OUTCOME_DIRS and _looks_like_episode(path)]
    for outcome in OUTCOME_DIRS:
        outcome_dir = root / outcome
        if outcome_dir.is_dir():
            candidates.extend((path for path in outcome_dir.iterdir() if _looks_like_episode(path)))
    return sorted(candidates, key=lambda path: path.name)

def _repo_id(output: Path) -> str:
    name = re.sub('[^A-Za-z0-9._-]+', '-', output.name).strip('-.') or 'pick-and-place'
    return f'mimic/{name}'

def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert('RGB'), dtype=np.uint8)
    if value.shape != (480, 640, 3):
        raise ValueError(f'unexpected image shape {value.shape}: {path}')
    return value

def convert(input_root: str | Path, output_dir: str | Path, *, task: str='pick_and_place', include_outcomes: Iterable[str]=('success',), allow_test_data: bool=False) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    source_root = Path(input_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if allow_test_data and ('_test' not in source_root.parts or '_test' not in output.parts):
        raise ValueError('--allow-test-data requires both input and output under explicit _test paths')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite processed dataset: {output}')
    candidates = discover_episodes(source_root)
    if not candidates:
        raise ValueError(f'no raw episodes found under {source_root}')
    allowed = set(include_outcomes)
    accepted: list[Path] = []
    accepted_schemas: set[str] = set()
    accepted_plan_seeds: set[tuple[str, int]] = set()
    weight_states: set[bool] = set()
    excluded: list[dict] = []
    for episode in candidates:
        validation = validate_mimic_raw(episode)
        expected_outcome = episode.parent.name if episode.parent.name in OUTCOME_DIRS else None
        if expected_outcome is not None and validation.get('outcome') != expected_outcome:
            raise ValueError(f"invalid raw episode {episode.name}: parent folder outcome '{expected_outcome}' does not match metadata outcome '{validation.get('outcome')}'")
        if not validation['valid']:
            raise ValueError(f"invalid raw episode {episode.name}: {'; '.join(validation['errors'])}")
        outcome = validation.get('outcome')
        if outcome not in allowed:
            excluded.append({'episode': episode.name, 'outcome': outcome, 'reason': 'outcome_not_selected'})
            continue
        metadata_path = episode / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
        test_exception = allow_test_data and metadata.get('test_only') is True and (metadata.get('production') is False)
        if outcome == 'success' and metadata.get('collection_accepted') is False and (not test_exception):
            excluded.append({'episode': episode.name, 'outcome': outcome, 'reason': 'collection_success_not_accepted'})
            continue
        plan_seed = (metadata.get('collection_plan_id'), metadata.get('distribution_seed'))
        if outcome == 'success' and plan_seed[0] is not None:
            if plan_seed in accepted_plan_seeds:
                raise ValueError(f'duplicate accepted success for {plan_seed[0]} seed {plan_seed[1]}: {episode.name}')
            accepted_plan_seeds.add(plan_seed)
        accepted.append(episode)
        accepted_schemas.add(validation['camera_schema'])
        with np.load(episode / 'samples_20hz.npz', allow_pickle=False) as raw:
            weight_states.add(validate_weight(raw, len(raw['action']), metadata) is not None)
    if not accepted:
        raise ValueError(f'no episodes matched allowed outcomes {sorted(allowed)}; excluded={excluded}')
    if len(accepted_schemas) != 1:
        raise ValueError('cannot mix raw camera schemas in one LeRobot dataset: ' + ', '.join(sorted(accepted_schemas)))
    if len(weight_states) != 1:
        raise ValueError('cannot mix operator-weight and legacy unlabelled episodes; use separate datasets')
    weight_enabled = next(iter(weight_states))
    camera_schema = next(iter(accepted_schemas))
    cameras = camera_names_for_schema(camera_schema)
    features = features_for_schema(camera_schema)
    if weight_enabled:
        features['weight'] = {'dtype': 'int64', 'shape': (1,), 'names': ['weight']}
    dataset = LeRobotDataset.create(repo_id=_repo_id(output), fps=FPS, features=features, root=output, robot_type='fr3', use_videos=False, image_writer_threads=4)
    source_records: list[dict] = []
    total_frames = 0
    try:
        for (episode_index, episode) in enumerate(accepted):
            with np.load(episode / 'samples_20hz.npz', allow_pickle=False) as raw:
                frame_count = int(raw['action'].shape[0])
                for frame_index in range(frame_count):
                    frame = {'observation.state': raw['observation_state'][frame_index].astype(np.float32, copy=True), 'action': raw['action'][frame_index].astype(np.float32, copy=True), 'task': task}
                    if weight_enabled:
                        frame['weight'] = np.asarray([raw['weight'][frame_index]], dtype=np.int64)
                    for camera in cameras:
                        frame[f'observation.images.{camera}'] = _rgb(episode / 'images' / camera / f'{frame_index:06d}.png')
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
                episode_metadata = json.loads((episode / 'metadata.json').read_text(encoding='utf-8'))
                source_record = {'episode_index': episode_index, 'directory': episode.name, 'frames': frame_count, 'outcome': episode_metadata.get('outcome'), 'initial_state_distribution': episode_metadata.get('initial_state_distribution'), 'distribution_seed': episode_metadata.get('distribution_seed')}
                if weight_enabled:
                    source_record['training_start_index'] = total_frames
                    source_record['weight_source'] = 'samples_20hz.npz:weight'
                source_records.append(source_record)
                total_frames += frame_count
        dataset.finalize()
    except BaseException:
        dataset.finalize()
        raise
    lerobot_root = Path(__import__('lerobot').__file__).resolve().parents[2]
    provenance = {'source_raw_root': str(source_root), 'source_episodes': source_records, 'excluded_episodes': excluded, 'episodes': len(source_records), 'frames': total_frames, 'task': task, 'raw_fps': FPS, 'camera_schema': camera_schema, 'temporal_convention': 'observation_t_before_step -> action_t', 'action_semantics': list(ACTION_NAMES), 'observation_features': [*(f'observation.images.{camera}' for camera in cameras), 'observation.state'], 'conversion_timestamp': datetime.now(timezone.utc).isoformat(), 'project_git_head': _git_head(PROJECT_ROOT), 'lerobot_git_head': _git_head(lerobot_root), 'lerobot_repo_id': dataset.repo_id, 'test_data_allowed': allow_test_data}
    provenance_path = output / 'meta' / 'mimic_conversion.json'
    if weight_enabled:
        provenance['sample_weight'] = dict(schema_version='operator_weight_v1', values=[1, 2, 3], feature='weight', policy_input=False, applied_to_loss=False, alignment='source sample i = CSV data row i = episode frame_index i; index = training_start_index + i')
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return provenance

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--task', default='pick_and_place')
    parser.add_argument('--include-outcome', action='append', default=None, choices=('success', 'failure', 'diagnostic'))
    parser.add_argument('--allow-test-data', action='store_true', help='Allow explicitly TEST-only records, only between _test paths')
    args = parser.parse_args()
    result = convert(args.input_root, args.output_dir, task=args.task, include_outcomes=args.include_outcome or ('success',), allow_test_data=args.allow_test_data)
    print(json.dumps(result, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
