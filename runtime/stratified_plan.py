"""Offline deterministic XY plans and freeze validation; never opens hardware."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIRECTORY = ROOT / 'experiments/mimicgen_square/configs/collection'
TRAIN_NAME = 'square_geomagic_xy50_v1'
EVAL_NAME = 'square_geomagic_xy_eval20_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fixed_context():
    from dataclasses import asdict
    from geomagic.teleop.input_deadband import DeadbandConfig
    from environments.mimicgen_square.config import (
        NUT_INITIAL_Z, SQUARE_WRIST_CAMERA_POS, SQUARE_WRIST_CAMERA_QUAT, SQUARE_WRIST_CAMERA_FOVY,
    )
    # Source seals detect changes to the fixed robot/task/cameras/controller.
    files = (
        'mimic/environments/mimicgen_square/scene.xml',
        'mimic/environments/mimicgen_square/config.py',
        'mimic/environments/mimicgen_square/task.py',
        'mimic/environments/mimic_robot/mimic/envs/assets/robots/franka_fr3/fr3.xml',
        'mimic/environments/mimicgen_square/assets/square_nut.xml',
        'geomagic/teleop/pose_mapper.py', 'geomagic/teleop/pipeline.py',
        'geomagic/teleop/fr3_target_controller.py', 'geomagic/teleop/robot_helpers.py',
        'omy_franka_teleop/launch/FR3_omy_bridge.py',
        'mimic/teleop/omy_fr3/launch/FR3_omy_bridge.py',
        'geomagic/leader/teleop_backend.py', 'mimic/runtime/ui_camera_layout.py',
        'geomagic/teleop/anti_windup.py', 'geomagic/teleop/gravity_support.py',
        'geomagic/teleop/input_deadband.py', 'geomagic/config/input_deadband.json',
        'geomagic/teleop/runtime_rotation.py', 'geomagic/standalone/haptics.py',
        'geomagic/standalone/haptic_config.py', 'geomagic/standalone/haptic_release.py',
        'geomagic/standalone/ui_haptics.py', 'geomagic/device/teleop_geo_force.h',
        'geomagic/device/teleop_geo_hdapi.cpp',
    )
    from geomagic.teleop.robot_helpers import BRIDGE_PATH
    import geomagic
    teleop_root = Path(geomagic.__file__).resolve().parent.parent
    def source_path(name):
        if name.startswith('mimic/'):
            return ROOT / name.removeprefix('mimic/')
        if name.startswith('omy_franka_teleop/'):
            return teleop_root / name.removeprefix('omy_franka_teleop/')
        return teleop_root / name
    hashes = {name: hashlib.sha256(source_path(name).read_bytes()).hexdigest()
              for name in files}
    return dict(object_z=float(NUT_INITIAL_Z),
        effective_configuration=dict(
            reference_mapping=os.environ.get('MIMIC_GEOMAGIC_REFERENCE_MAPPING') == '1',
            deadband=asdict(DeadbandConfig.from_environment()),
            haptics_enabled=os.environ.get('MIMIC_GEOMAGIC_HAPTICS', '1') == '1'),
        wrist_camera=dict(pos=list(SQUARE_WRIST_CAMERA_POS), quat=list(SQUARE_WRIST_CAMERA_QUAT),
                          fovy=SQUARE_WRIST_CAMERA_FOVY), source_sha256=hashes)


def validate_config(config, *, evaluation=False):
    expected = r'square_geomagic_xy_eval20_v[1-9][0-9]*' if evaluation else r'square_geomagic_xy50_v[1-9][0-9]*'
    if not re.fullmatch(expected, str(config.get('plan_name', ''))):
        raise ValueError('invalid plan version/name')
    grid = config['grid']
    wanted = (4, 5, 1) if evaluation else (5, 5, 2)
    if (any(type(grid.get(k)) is not int for k in ('rows','cols','samples_per_cell'))
            or tuple(grid.get(k) for k in ('rows', 'cols', 'samples_per_cell')) != wanted):
        raise ValueError(f'grid must be rows/cols/samples_per_cell={wanted}')
    if any(type(config.get(k)) is not int or config[k] < 0 for k in ('seed', 'shuffle_seed')):
        raise ValueError('seed and shuffle_seed must be nonnegative integers')
    margin = config['cell_sampling_margin_fraction']
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin < .5:
        raise ValueError('cell_sampling_margin_fraction must be finite in [0, 0.5)')
    bounds = config['workspace']
    if set(bounds) != {'x_min', 'x_max', 'y_min', 'y_max'}:
        raise ValueError('workspace must use x_min/x_max/y_min/y_max in meters')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in bounds.values()):
        raise ValueError('workspace bounds must be finite numbers')
    if bounds['x_min'] >= bounds['x_max'] or bounds['y_min'] >= bounds['y_max']:
        raise ValueError('workspace must have positive width and height')
    yaw = config['object_yaw_world_rad']
    if type(yaw) not in (int, float) or not math.isfinite(yaw):
        raise ValueError('fixed object yaw must be finite')
    if config['object_z'] != fixed_context()['object_z']:
        raise ValueError('object Z must equal the existing Square task reset Z')
    if not evaluation and config['default_raw_root'] != (
            'mimicgen_square/datasets/raw/restricted30_geomagic/' + config['plan_name']):
        raise ValueError('XY50 must use its separate plan-version subdirectory under the Geomagic root')


def generate_states(config, *, exclude=()):
    bounds, grid = config['workspace'], config['grid']
    dx = (bounds['x_max'] - bounds['x_min']) / grid['cols']
    dy = (bounds['y_max'] - bounds['y_min']) / grid['rows']
    margin = config['cell_sampling_margin_fraction']
    rng = random.Random(config['seed'])
    occupied = set(exclude)
    states = []
    for row in range(grid['rows']):
        for col in range(grid['cols']):
            cell = dict(x_min=bounds['x_min'] + col * dx,
                        x_max=bounds['x_min'] + (col + 1) * dx,
                        y_min=bounds['y_min'] + row * dy,
                        y_max=bounds['y_min'] + (row + 1) * dy)
            sampling = dict(x_min=cell['x_min'] + margin * dx,
                            x_max=cell['x_max'] - margin * dx,
                            y_min=cell['y_min'] + margin * dy,
                            y_max=cell['y_max'] - margin * dy)
            if any(math.nextafter(sampling[f'{axis}_min'], math.inf) >=
                   math.nextafter(sampling[f'{axis}_max'], -math.inf) for axis in ('x', 'y')):
                raise ValueError('cell sampling area is too small for distinct interior floating-point XY')
            for sample_index in range(grid['samples_per_cell']):
                for _ in range(1000):
                    # Open endpoints even when the user explicitly selects zero margin.
                    x = rng.uniform(math.nextafter(sampling['x_min'], sampling['x_max']),
                                    math.nextafter(sampling['x_max'], sampling['x_min']))
                    y = rng.uniform(math.nextafter(sampling['y_min'], sampling['y_max']),
                                    math.nextafter(sampling['y_max'], sampling['y_min']))
                    if (x, y) not in occupied:
                        break
                else:
                    raise ValueError('unable to generate a distinct finite-precision XY')
                occupied.add((x, y))
                states.append(dict(plan_index=len(states),
                    planned_state_id=f'{config["plan_name"]}:r{row}c{col}s{sample_index}',
                    cell_id=f'cell_{row}{col}', cell_row=row, cell_col=col,
                    sample_in_cell=sample_index, cell_bounds=cell,
                    sampling_bounds=sampling, x=x, y=y, status='planned'))
    order = list(range(len(states)))
    random.Random(config['shuffle_seed']).shuffle(order)
    for collection_order, index in enumerate(order):
        states[index]['collection_order'] = collection_order
    return states


def expected_manifest(config, *, exclude=(), parent_sha256=None):
    evaluation = 'training_plan' in config
    validate_config(config, evaluation=evaluation)
    content = dict(plan_name=config['plan_name'], configuration=deepcopy(config),
        seed=config['seed'], shuffle_seed=config['shuffle_seed'], workspace=config['workspace'],
        grid=config['grid'], object_z=config['object_z'], object_yaw_world_rad=config['object_yaw_world_rad'],
        cell_sampling_margin_fraction=config['cell_sampling_margin_fraction'],
        fixed_context=fixed_context(), training_plan_sha256=parent_sha256,
        states=generate_states(config, exclude=exclude))
    return dict(status='DRAFT', content_sha256=digest(content), **content)


def paths(name, directory=None):
    directory = Path(directory or PLAN_DIRECTORY)
    if not re.fullmatch(r'square_geomagic_xy(?:50|_eval20)_v[1-9][0-9]*', name):
        raise ValueError('invalid plan name')
    return directory / f'{name}.config.json', directory / f'{name}.plan.json'


def expected_pair(name=TRAIN_NAME, directory=None):
    config_path, _ = paths(name, directory)
    config = read_json(config_path)
    if config['plan_name'] != name:
        raise ValueError('training config name must match its filename')
    train = expected_manifest(config)
    eval_name = config['evaluation_plan']
    eval_config = read_json(paths(eval_name, directory)[0])
    if eval_config['plan_name'] != eval_name:
        raise ValueError('evaluation config name must match its filename')
    if eval_config['training_plan'] != name or eval_config['seed'] == config['seed']:
        raise ValueError('evaluation must reference training and use a different seed')
    # The only workspace definition lives in training config.
    if 'workspace' in eval_config:
        raise ValueError('evaluation workspace must be inherited, not independently configured')
    if eval_config['cell_sampling_margin_fraction'] != config['cell_sampling_margin_fraction']:
        raise ValueError('evaluation must use the same per-cell margin fraction as training')
    eval_config.update(workspace=deepcopy(config['workspace']), object_z=config['object_z'],
                       object_yaw_world_rad=config['object_yaw_world_rad'])
    evaluation = expected_manifest(eval_config,
        exclude={(s['x'], s['y']) for s in train['states']}, parent_sha256=train['content_sha256'])
    return train, evaluation


def seal_path(name, directory=None):
    return Path(directory or PLAN_DIRECTORY) / f'{name}.frozen.json'


def load_manifest(name=TRAIN_NAME, directory=None, *, require_frozen=False, allow_missing_seal=False):
    config = read_json(paths(name, directory)[0])
    training_name = config.get('training_plan', name)
    expected = expected_pair(training_name, directory)
    selected = next((doc for doc in expected if doc['plan_name'] == name), None)
    if selected is None:
        raise ValueError('plan is not part of its declared training/evaluation pair')
    document = read_json(paths(name, directory)[1])
    wanted = dict(selected, status=document.get('status'))
    if wanted['status'] not in ('DRAFT', 'FROZEN') or document != wanted:
        raise ValueError('plan/config/fixed-context mismatch; regenerate DRAFT explicitly or create v2 for FROZEN')
    seal = seal_path(training_name, directory)
    if seal.exists():
        seals = read_json(seal)
        if document['status'] != 'FROZEN' or seals.get(name) != document['content_sha256']:
            raise ValueError('immutable frozen-plan seal mismatch')
    elif document['status'] == 'FROZEN' and not allow_missing_seal:
        raise ValueError('frozen plan is missing its seal; retry freeze with unchanged configuration')
    if require_frozen and document['status'] != 'FROZEN':
        raise ValueError('collection blocked: plan is DRAFT; inspect workspace and freeze first')
    if document['status'] == 'FROZEN':
        from .workspace_validation import require_approval
        require_approval(training_name, directory, configuration=expected[0]['configuration'])
    return document


def generate_pair(name=TRAIN_NAME, directory=None):
    expected = expected_pair(name, directory)
    if seal_path(name, directory).exists():
        # Validate and return; never overwrite a frozen version, even unchanged.
        return tuple(load_manifest(doc['plan_name'], directory) for doc in expected)
    from .workspace_validation import status_path, require_approval
    if status_path(name, directory).exists():
        require_approval(name, directory, configuration=expected[0]['configuration'])
    for document in expected:
        path = paths(document['plan_name'], directory)[1]
        if path.exists() and read_json(path).get('status') == 'FROZEN':
            raise ValueError('refusing to regenerate FROZEN; retry freeze or create v2')
    for document in expected:
        write_json(paths(document['plan_name'], directory)[1], document)
    return expected


def freeze_pair(name=TRAIN_NAME, directory=None):
    expected = expected_pair(name, directory)
    from .workspace_validation import require_approval
    require_approval(name, directory, configuration=expected[0]['configuration'])
    # Explicit mismatch errors if DRAFT configuration changed after inspection.
    for document in expected:
        load_manifest(document['plan_name'], directory, allow_missing_seal=True)
    if seal_path(name, directory).exists():
        return tuple(load_manifest(doc['plan_name'], directory, require_frozen=True) for doc in expected)
    frozen = tuple(dict(doc, status='FROZEN') for doc in expected)
    for document in frozen:
        write_json(paths(document['plan_name'], directory)[1], document)
    # Exclusive create, never replace the immutable seal on a later freeze.
    with seal_path(name, directory).open('x') as stream:
        json.dump({doc['plan_name']: doc['content_sha256'] for doc in frozen}, stream, indent=2)
        stream.write('\n')
    return frozen


def feasibility(document):
    w = document['workspace']
    x0,x1,y0,y1 = (w[k] for k in ('x_min','x_max','y_min','y_max'))
    xc,yc = (x0+x1)/2,(y0+y1)/2
    positions = {'center':(xc,yc), 'front-left':(x1,y1), 'front-right':(x1,y0),
                 'back-left':(x0,y1), 'back-right':(x0,y0),
                 'front-mid':(x1,yc), 'back-mid':(x0,yc),
                 'left-mid':(xc,y1), 'right-mid':(xc,y0)}
    return dict(plan_name=document['plan_name'], dataset_states=False,
        convention='front=+X, left=+Y; rows increase +Y, columns increase +X',
        object_z=document['object_z'], object_yaw_world_rad=document['object_yaw_world_rad'],
        positions={key:list(value) for key,value in positions.items()})


def coverage_summary(document):
    """Validate coverage offline, without enforcing extra sample spacing."""
    from collections import Counter
    states, grid, workspace = document['states'], document['grid'], document['workspace']
    rows, cols, per_cell = (grid[k] for k in ('rows', 'cols', 'samples_per_cell'))
    counts = Counter((s['cell_row'], s['cell_col']) for s in states)
    expected_cells = {(r, c) for r in range(rows) for c in range(cols)}
    total = rows * cols * per_cell
    if (len(states) != total or set(counts) != expected_cells
            or any(count != per_cell for count in counts.values())
            or len({s['planned_state_id'] for s in states}) != total
            or sorted(s['plan_index'] for s in states) != list(range(total))
            or sorted(s['collection_order'] for s in states) != list(range(total))
            or len({(s['x'], s['y']) for s in states}) != total):
        raise ValueError('plan has invalid cell coverage, identities, order, or duplicate coordinates')
    for state in states:
        if state['cell_id'] != f"cell_{state['cell_row']}{state['cell_col']}":
            raise ValueError('cell ID does not match row and column')
        for area in (workspace, state['cell_bounds'], state['sampling_bounds']):
            if not all(area[f'{a}_min'] < state[a] < area[f'{a}_max'] for a in ('x', 'y')):
                raise ValueError('sample is outside its workspace/cell/sampling interior')
    width, height = workspace['x_max']-workspace['x_min'], workspace['y_max']-workspace['y_min']
    margin = document['cell_sampling_margin_fraction']
    return dict(plan_name=document['plan_name'], status=document['status'], total_states=total,
        cells=len(counts), samples_per_cell=per_cell, workspace_width_m=width,
        workspace_height_m=height, cell_width_m=width/cols, cell_height_m=height/rows,
        per_side_margin_m=[margin*width/cols, margin*height/rows],
        sampling_width_per_cell_m=(1-2*margin)*width/cols,
        sampling_height_per_cell_m=(1-2*margin)*height/rows,
        workspace_area_m2=width*height, effective_sampling_area_m2=width*height*(1-2*margin)**2,
        minimum_pairwise_distance_m=min(math.hypot(a['x']-b['x'], a['y']-b['y'])
            for i, a in enumerate(states) for b in states[i+1:]))


def plot_plan(document, output, *, accepted=(), failed=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    w,g = document['workspace'],document['grid']
    fig,ax = plt.subplots(figsize=(13,4))
    ax.add_patch(Rectangle((w['x_min']*1000,w['y_min']*1000),
        (w['x_max']-w['x_min'])*1000,(w['y_max']-w['y_min'])*1000,fill=False,color='black'))
    for col in range(1,g['cols']):
        ax.axvline((w['x_min']+(w['x_max']-w['x_min'])*col/g['cols'])*1000,color='gray',lw=.6)
    for row in range(1,g['rows']):
        ax.axhline((w['y_min']+(w['y_max']-w['y_min'])*row/g['rows'])*1000,color='gray',lw=.6)
    for row in range(g['rows']):
        for col in range(g['cols']):
            ax.text((w['x_min']+(col+.5)*(w['x_max']-w['x_min'])/g['cols'])*1000,
                    (w['y_min']+(row+.08)*(w['y_max']-w['y_min'])/g['rows'])*1000,
                    f'cell_{row}{col}',fontsize=6,ha='center',va='bottom')
    groups = [('planned','o','#557eb8',24),('accepted','s','#20985d',38),
              ('failed/retried','x','#db7c20',48)]
    for label,marker,color,size in groups:
        subset=[s for s in document['states'] if
            (label=='planned' or label=='accepted' and s['collection_order'] in accepted
             or label=='failed/retried' and s['collection_order'] in failed)]
        ax.scatter([s['x']*1000 for s in subset],[s['y']*1000 for s in subset],
                   marker=marker,c=color,s=size,label=label,alpha=.8)
    validation = document.get('validation_points', [])
    if validation:
        ax.scatter([p['x']*1000 for p in validation], [p['y']*1000 for p in validation],
                   marker='D', c='#8944ad', s=38, label='9 manual PASS points')
        for point in validation:
            ax.annotate(point['name'],(point['x']*1000,point['y']*1000),
                        xytext=(4,4),textcoords='offset points',fontsize=6)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('World X [mm]');ax.set_ylabel('World Y [mm]')
    ax.set_title(f'{document["plan_name"]} · {document["status"]} · {len(document["states"])} states')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.4),ncol=3)
    fig.tight_layout();fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)


def render_plot(document, output, *, accepted=(), failed=(), python=None):
    """Optional separate plotting interpreter; validated JSON only, no hardware."""
    if python is None:
        return plot_plan(document, output, accepted=accepted, failed=failed)
    payload = dict(document=document, accepted=list(accepted), failed=list(failed))
    env = dict(os.environ)
    # Child imports only this module and matplotlib, never the task/config.
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([python, '-B', '-c',
        'import json, sys; from runtime.stratified_plan import plot_plan; '
        'p=json.load(sys.stdin); plot_plan(p["document"], sys.argv[1], '
        'accepted=p["accepted"], failed=p["failed"])', str(output)],
        input=json.dumps(payload), text=True, env=env, check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('generate','freeze','feasibility','plot','validate'))
    parser.add_argument('--plan',default=TRAIN_NAME)
    parser.add_argument('--directory',type=Path,default=PLAN_DIRECTORY)
    parser.add_argument('--confirm-workspace',action='store_true')
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--no-plot',action='store_true',help='generate JSON without optional matplotlib')
    parser.add_argument('--plot-python',help='optional matplotlib interpreter, e.g. /usr/bin/python3')
    args=parser.parse_args()
    if args.command=='generate':
        documents=generate_pair(args.plan,args.directory)
        for document in documents:
            coverage_summary(document)
            if not args.no_plot:
                render_plot(document,args.directory/f'{document["plan_name"]}.png',python=args.plot_python)
        write_json(args.directory/f'{args.plan}.feasibility.json',feasibility(documents[0]))
        print(json.dumps({'plans':[{k:d[k] for k in ('plan_name','status','content_sha256')} for d in documents]},indent=2))
    elif args.command=='freeze':
        if not args.confirm_workspace:
            parser.error('freeze requires --confirm-workspace after manual workspace validation')
        print(json.dumps([{k:d[k] for k in ('plan_name','status')} for d in freeze_pair(args.plan,args.directory)],indent=2))
    elif args.command=='feasibility':
        print(json.dumps(feasibility(load_manifest(args.plan,args.directory)),indent=2))
    elif args.command=='validate':
        print(json.dumps(coverage_summary(load_manifest(args.plan,args.directory)),indent=2))
    else:
        document=load_manifest(args.plan,args.directory)
        accepted,failed=set(),set()
        if args.raw_root:
            from .collection_plan import XYCollectionPlan
            plan=XYCollectionPlan.from_document(document,args.directory)
            accepted=plan.accepted_seeds(args.raw_root)
            failed=plan.failed_states(args.raw_root)
        output=args.directory/f'{args.plan}.coverage.png'
        render_plot(document,output,accepted=accepted,failed=failed,python=args.plot_python)
        print(output)


if __name__=='__main__': main()
