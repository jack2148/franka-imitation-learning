"""Deterministic task identities and experiment workspace defaults."""
from __future__ import annotations
from dataclasses import dataclass, replace
import os
import re
from runtime.collection_workspace import research_root
TASK_ALIASES = {'fr3-pick-place': 'pick_and_place', 'pick_and_place': 'pick_and_place', 'mimicgen_square': 'mimicgen_square', 'fr3-mimicgen-square': 'mimicgen_square'}

@dataclass(frozen=True)
class TaskProfile:
    task: str
    experiment: str
    dataset_version: str
    collection_plan_id: str | None = None
PROFILES = {'pick_and_place': TaskProfile(task='pick_and_place', experiment='pick_and_place', dataset_version='omy_fr3_operator_wrist_20hz_v1'), 'mimicgen_square': TaskProfile(task='mimicgen_square', experiment='mimicgen_square', dataset_version='omy_fr3_mimicgen_square_operator_wrist_20hz_v1', collection_plan_id='square_restricted30_v1')}

def canonical_task(value: str) -> str:
    if value not in {'fr3-pick-place', 'pick_and_place', 'mimicgen_square', 'fr3-mimicgen-square'}:
        raise ValueError('This public snapshot includes Square and the base Pick-and-Place environment; see docs/PUBLIC_V1_SCOPE.md for extensions')
    try:
        return TASK_ALIASES[value]
    except KeyError as error:
        raise ValueError(f'unsupported task: {value}') from error

def launch_leader() -> str:
    """Leader identity is independent of task; OMY remains the default."""
    value = os.environ.get('MIMIC_UI_LEADER', 'omy')
    if value not in {'omy', 'geomagic'}:
        raise ValueError('public leader must be omy or geomagic')
    return value
GEOMAGIC_SOURCES = ('synthetic_udp', 'physical_hdapi')

def launch_geomagic_source() -> str | None:
    """Geomagic device source; None for OMY. Synthetic remains the default."""
    if launch_leader() != 'geomagic':
        return None
    value = os.environ.get('MIMIC_UI_GEOMAGIC_SOURCE') or 'synthetic_udp'
    if value not in GEOMAGIC_SOURCES:
        raise ValueError('Geomagic source must be synthetic_udp or physical_hdapi')
    return value

def launch_profile() -> TaskProfile:
    """Return launcher-selected identity; generic starts default to Pick-and-Place."""
    profile = PROFILES[canonical_task(os.environ.get('MIMIC_UI_TASK', 'pick_and_place'))]
    if research_root() is not None:
        if os.environ.get('MIMIC_WEIGHT_COLLECTION') == '1':
            if launch_leader() != 'geomagic' or profile.task != 'mimicgen_square':
                raise ValueError('Named weight collection requires Touch Square')
            return replace(profile, dataset_version='geo_sq_weight_v1', collection_plan_id=None)
        raise ValueError('Public v1 collection workspace requires the ordinary Touch launcher')
    if launch_leader() == 'geomagic':
        selected = profile.collection_plan_id
        if profile.task == 'mimicgen_square' and launch_geomagic_source() == 'physical_hdapi':
            validation = os.environ.get('MIMIC_UI_WORKSPACE_VALIDATION')
            if validation:
                if not re.fullmatch('square_geomagic_xy50_v[1-9][0-9]*', validation):
                    raise ValueError('invalid workspace validation plan')
                return replace(profile, dataset_version=profile.dataset_version.replace('omy_', 'geomagic_', 1), collection_plan_id=None)
            if not os.environ.get('MIMIC_UI_COLLECTION_PLAN'):
                from runtime.stratified_plan import seal_path, TRAIN_NAME
                if seal_path(TRAIN_NAME).exists():
                    selected = TRAIN_NAME
            selected = os.environ.get('MIMIC_UI_COLLECTION_PLAN') or selected
            if selected != 'square_restricted30_v1' and (not re.fullmatch('square_geomagic_xy50_v[1-9][0-9]*', selected)):
                raise ValueError('physical Square collection plan must be Restricted30 or a versioned XY50 training plan')
        return replace(profile, dataset_version=profile.dataset_version.replace('omy_', 'geomagic_', 1), collection_plan_id=selected)
    return profile

def purpose_defaults(profile: TaskProfile | None=None) -> dict[str, str]:
    selected = profile or launch_profile()
    if os.environ.get('MIMIC_WEIGHT_COLLECTION') == '1':
        from pathlib import Path
        home = str(Path.home())
        return dict(dataset_output=home, raw_input=home, processed_output=home, training_input=home, training_output=home)
    if research_root() is not None:
        suffix = '/_test' if launch_geomagic_source() == 'synthetic_udp' else ''
        return dict(dataset_output='raw' + suffix, raw_input='raw' + suffix, processed_output='processed' + suffix, training_input='processed' + suffix, training_output='training' + suffix)
    experiment = selected.experiment
    raw_root = f'{experiment}/datasets/raw'
    if launch_leader() == 'geomagic' and launch_geomagic_source() == 'physical_hdapi' and (selected.task == 'mimicgen_square') and os.environ.get('MIMIC_UI_WORKSPACE_VALIDATION'):
        raw_root = f"{experiment}/analysis/workspace_validation/{os.environ['MIMIC_UI_WORKSPACE_VALIDATION']}"
        return dict(dataset_output=raw_root, raw_input=raw_root, processed_output=f'{experiment}/datasets/processed', training_input=f'{experiment}/datasets/processed', training_output=f'{experiment}/training')
    if selected.collection_plan_id is not None:
        from runtime.collection_plan import collection_plan
        raw_root = collection_plan(selected.collection_plan_id).default_raw_root
    if launch_leader() == 'geomagic' and (not (selected.collection_plan_id or '').startswith('square_geomagic_xy50_v')):
        suffix = 'geomagic' if launch_geomagic_source() == 'physical_hdapi' else 'geomagic_synthetic'
        raw_root += f'_{suffix}' if selected.collection_plan_id else f'/{suffix}'
    return {'dataset_output': raw_root, 'raw_input': raw_root, 'processed_output': f'{experiment}/datasets/processed', 'training_input': f'{experiment}/datasets/processed', 'training_output': f'{experiment}/training'}
