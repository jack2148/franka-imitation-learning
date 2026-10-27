"""Manual nine-point assessment. No motion or PASS result is automated."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from . import stratified_plan as sp

CANDIDATE = dict(x_min=.445, x_max=.485, y_min=-.1425, y_max=-.0825)
POINTS = (
    ('center', .4650, -.1125), ('front-left', .4850, -.0825),
    ('front-right', .4850, -.1425), ('back-left', .4450, -.0825),
    ('back-right', .4450, -.1425), ('front-mid', .4850, -.1125),
    ('back-mid', .4450, -.1125), ('left-mid', .4650, -.0825),
    ('right-mid', .4650, -.1425),
)
CATEGORIES = ('initial_collision', 'camera_occlusion', 'unreachable', 'joint_limit',
    'joint_rate_saturation', 'controller_contact_lock', 'grasp_difficulty',
    'operator_ergonomics', 'other')


def status_path(name=sp.TRAIN_NAME, directory=None):
    sp.paths(name, directory)  # Reject path traversal / invalid versions.
    return Path(directory or sp.PLAN_DIRECTORY) / f'{name}.workspace_validation.json'


def candidate_config(name=sp.TRAIN_NAME, directory=None):
    config = sp.read_json(sp.paths(name, directory)[0])
    config.pop('workspace_validation_sha256', None)
    config.update(workspace=deepcopy(CANDIDATE),
                  workspace_source='Nine-point manual Geomagic validation candidate: 40 x 60 mm')
    return config


def create_validation(name=sp.TRAIN_NAME, directory=None):
    path = status_path(name, directory)
    if path.exists():
        return load_validation(name, directory)
    if sp.seal_path(name, directory).exists():
        raise ValueError('cannot create new validation for a frozen version; create v2')
    config = candidate_config(name, directory)
    document = dict(plan_name=name, workspace=deepcopy(CANDIDATE), workspace_status='DRAFT',
        object_z=config['object_z'], object_yaw_world_rad=config['object_yaw_world_rad'],
        fixed_context=sp.fixed_context(), training_settings_sha256=sp.digest(config),
        evaluation_settings_sha256=sp.digest(sp.read_json(sp.paths(config['evaluation_plan'],directory)[0])),
        current_point='center', convention='front=+X; left=+Y',
        criteria=['initial_state', 'visibility', 'reachability', 'manipulation', 'operator_ergonomics'],
        points=[dict(name=n, x=x, y=y, status='pending', categories=[], notes='',
                     attempt_number=1, history=[]) for n,x,y in POINTS])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids overwriting a concurrently recorded assessment.
    with path.open('x') as stream:
        json.dump(document, stream, indent=2); stream.write('\n')
    return document


def load_validation(name=sp.TRAIN_NAME, directory=None):
    document = sp.read_json(status_path(name, directory))
    config = candidate_config(name, directory)
    if (document.get('plan_name') != name or document.get('workspace') != CANDIDATE
            or document.get('workspace_status') not in ('DRAFT', 'FROZEN')
            or document.get('object_z') != config['object_z']
            or document.get('object_yaw_world_rad') != config['object_yaw_world_rad']
            or document.get('fixed_context') != sp.fixed_context()
            or document.get('training_settings_sha256') != sp.digest(config)
            or document.get('evaluation_settings_sha256') != sp.digest(
                sp.read_json(sp.paths(config['evaluation_plan'],directory)[0]))):
        raise ValueError('workspace validation/config/fixed-context mismatch; assessments cannot be reused')
    points = document.get('points', [])
    if len(points) != 9 or document.get('current_point') not in {n for n,_,_ in POINTS}:
        raise ValueError('validation must contain the exact nine required points')
    for point, (n,x,y) in zip(points, POINTS):
        if (point.get('name') != n or point.get('x') != x or point.get('y') != y
                or point.get('status') not in ('pending','pass','fail')
                or type(point.get('attempt_number')) is not int or point['attempt_number'] < 1
                or not isinstance(point.get('notes'), str)
                or not isinstance(point.get('categories'), list)
                or any(c not in CATEGORIES for c in point['categories'])
                or not isinstance(point.get('history'), list)
                or point['status'] == 'fail' and (not point['notes'].strip() or not point['categories'])):
            raise ValueError('invalid point identity, assessment, failure reason or attempt number')
    return document


@contextmanager
def locked(name, directory):
    path = status_path(name, directory).with_suffix('.lock')
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def update_validation(action, name=sp.TRAIN_NAME, directory=None, *, point=None,
                      status=None, categories=(), notes=''):
    with locked(name, directory):
        document = load_validation(name, directory)
        if document['workspace_status'] == 'FROZEN' or sp.seal_path(name, directory).exists():
            raise ValueError('frozen validation is immutable; create v2 for changes')
        selected = point or document['current_point']
        record = next((p for p in document['points'] if p['name'] == selected), None)
        if record is None: raise ValueError('unknown validation point')
        if action == 'select':
            document['current_point'] = selected
        elif action == 'next':
            index = next(i for i,p in enumerate(document['points']) if p['name']==selected)
            if record['status'] == 'pending': raise ValueError('assess this point or select another explicitly')
            if index == 8: raise ValueError('already at point 9/9; inspect summary')
            document['current_point'] = document['points'][index+1]['name']
        elif action in ('mark', 'retry'):
            if action == 'mark':
                if status not in ('pass','fail'): raise ValueError('assessment must be pass or fail')
                if any(c not in CATEGORIES for c in categories): raise ValueError('unknown failure category')
                if status == 'fail' and (not categories or not notes.strip()):
                    raise ValueError('FAIL requires a category and a free-text reason')
            record['history'].append(dict(action=action, status=status if action=='mark' else 'pending',
                categories=list(categories), notes=notes, attempt_number=record['attempt_number'],
                at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            if action == 'retry':
                document['current_point'] = selected
                record['attempt_number'] += 1
                record.update(status='pending', categories=[], notes='')
            else:
                record.update(status=status, categories=list(categories), notes=notes)
        else: raise ValueError('unknown validation action')
        sp.write_json(status_path(name, directory), document)
        return summary(document)


def summary(document):
    current = next(p for p in document['points'] if p['name']==document['current_point'])
    index = document['points'].index(current)+1
    passed = sum(p['status']=='pass' for p in document['points'])
    failures = [p for p in document['points'] if p['status']=='fail']
    recommendations = []
    for bound, names in (
        ('x_max', {'front-left','front-right','front-mid'}),
        ('x_min', {'back-left','back-right','back-mid'}),
        ('y_max', {'front-left','back-left','left-mid'}),
        ('y_min', {'front-right','back-right','right-mid'})):
        relevant = [p['name'] for p in failures if p['name'] in names]
        if relevant: recommendations.append(dict(bound=bound, failed_points=relevant,
            recommendation='review this boundary and recorded reasons; do not automatically shrink'))
    if any(p['name']=='center' for p in failures):
        recommendations.append(dict(bound=None, failed_points=['center'],
            recommendation='review task setup/visibility/controller/ergonomics before changing boundaries'))
    recommendations.sort(key=lambda r:len(r['failed_points']),reverse=True)
    return dict(plan_name=document['plan_name'], workspace=document['workspace'],
        workspace_status=document['workspace_status'], point=current, index=index, total=9,
        passed=passed, failed=len(failures), pending=9-passed-len(failures),
        can_freeze=passed==9, points=document['points'], boundary_review=recommendations,
        recording_enabled=False)


def approval_digest(document):
    return sp.digest({key:document[key] for key in ('plan_name','workspace','object_z',
        'object_yaw_world_rad','fixed_context','training_settings_sha256','evaluation_settings_sha256','points')})


def require_approval(name=sp.TRAIN_NAME, directory=None, *, configuration=None):
    try:
        document = load_validation(name, directory)
    except FileNotFoundError as error:
        raise ValueError('freeze blocked: nine-point manual validation file is required') from error
    if any(p['status'] != 'pass' for p in document['points']):
        raise ValueError('freeze blocked: all nine manually assessed points must PASS (9/9)')
    if configuration is not None and (configuration['workspace'] != document['workspace']
            or configuration.get('workspace_validation_sha256') != approval_digest(document)):
        raise ValueError('freeze/configuration does not match approved workspace and assessment checksum')
    return document


def freeze_validated(name=sp.TRAIN_NAME, directory=None, *, plot_python=None, plots=True):
    directory = Path(directory or sp.PLAN_DIRECTORY)
    with locked(name, directory):
        validation = require_approval(name, directory)
        first_freeze = not sp.seal_path(name, directory).exists()
        if first_freeze:
            # Preserve the old bad draft before any active config/plan changes.
            archive = directory/'superseded_60x5_draft'
            archive.mkdir(exist_ok=True)
            evaluation = candidate_config(name, directory)['evaluation_plan']
            for plan in (name, evaluation):
                for source in directory.glob(f'{plan}.*'):
                    if 'workspace_validation' in source.name or source.suffix == '.lock': continue
                    target = archive/source.name
                    if not target.exists(): shutil.copy2(source, target)
            (archive/'SUPERSEDED.md').write_text('Superseded 60 x 5 mm DRAFT evidence. Do not use for collection.\n')
            config = candidate_config(name, directory)
            config['workspace_validation_sha256'] = approval_digest(validation)
            sp.write_json(sp.paths(name, directory)[0], config)
            existing_plans = [sp.paths(plan,directory)[1] for plan in (name,evaluation)]
            if not any(path.exists() and sp.read_json(path).get('status')=='FROZEN' for path in existing_plans):
                sp.generate_pair(name, directory)
        accounting = None
        if first_freeze and directory.resolve() == sp.PLAN_DIRECTORY.resolve():
            from .collection_plan import XYCollectionPlan
            plan = XYCollectionPlan.from_document(sp.expected_pair(name,directory)[0], directory)
            raw_root = sp.ROOT/'mimic/experiments'/plan.default_raw_root
            plan.ensure_isolated_root(raw_root)
            if plan.accepted_seeds(raw_root):
                raise ValueError('new plan already contains accepted data; accounting was not reset')
            ledger = plan._attempt_ledger(raw_root)
            accounting = (raw_root, ledger)
        documents = sp.freeze_pair(name, directory)
        validation['workspace_status'] = 'FROZEN'
        sp.write_json(status_path(name, directory), validation)
        # Only the real, explicitly approved freeze initializes production
        # accounting. Unit tests with alternate directories never touch it.
        if accounting is not None:
            raw_root, ledger = accounting
            raw_root.mkdir(parents=True, exist_ok=True)
            if not (raw_root/'.xy50-attempts.json').exists():
                sp.write_json(raw_root/'.xy50-attempts.json', ledger)
        if plots:
            overlay = [dict(name=p['name'],x=p['x'],y=p['y']) for p in validation['points']]
            for doc in documents:
                sp.render_plot(dict(doc,validation_points=overlay),directory/f'{doc["plan_name"]}.png',
                               python=plot_python)
            sp.write_json(directory/f'{name}.feasibility.json', sp.feasibility(documents[0]))
        return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='launch',
        choices=('launch','select','mark','retry','next','summary','freeze'))
    parser.add_argument('--plan', default=sp.TRAIN_NAME)
    parser.add_argument('--directory', type=Path, default=sp.PLAN_DIRECTORY)
    parser.add_argument('--point', choices=[n for n,_,_ in POINTS])
    parser.add_argument('--status', choices=('pass','fail'))
    parser.add_argument('--category', choices=CATEGORIES, action='append', default=[])
    parser.add_argument('--note', default='')
    parser.add_argument('--plot-python', default='/usr/bin/python3')
    args = parser.parse_args()
    if args.command == 'launch':
        if args.directory.resolve() != sp.PLAN_DIRECTORY.resolve():
            parser.error('physical launch only supports the canonical plan directory')
        create_validation(args.plan,args.directory)
        if args.point: update_validation('select',args.plan,args.directory,point=args.point)
        if sp.seal_path(args.plan,args.directory).exists(): parser.error('validation is frozen; use collection UI')
        launcher = sp.ROOT/'mimic/scripts/ui/ui_geo_sq'
        if not launcher.exists(): parser.error(f'physical launcher missing: {launcher}')
        env = dict(os.environ, MIMIC_UI_WORKSPACE_VALIDATION=args.plan,
                   MIMIC_GEOMAGIC_MEASURE_ONLY='1', GEOMAGIC_SYNC_AUTO='0')
        env.pop('MIMIC_UI_COLLECTION_PLAN',None); env.pop('GEOMAGIC_SYNC_CAPTURE',None)
        os.execve(str(launcher),[str(launcher)],env)
    elif args.command == 'summary':
        print(json.dumps(summary(load_validation(args.plan,args.directory)),indent=2))
    elif args.command == 'freeze':
        documents = freeze_validated(args.plan,args.directory,plot_python=args.plot_python)
        print(json.dumps([dict(plan_name=d['plan_name'],status=d['status'],
            total=len(d['states']),x_min=min(s['x'] for s in d['states']),
            x_max=max(s['x'] for s in d['states']),y_min=min(s['y'] for s in d['states']),
            y_max=max(s['y'] for s in d['states'])) for d in documents],indent=2))
    else:
        if args.command=='select' and args.point is None: parser.error('select requires --point')
        print(json.dumps(update_validation(args.command,args.plan,args.directory,point=args.point,
            status=args.status,categories=args.category,notes=args.note),indent=2))


if __name__ == '__main__': main()
