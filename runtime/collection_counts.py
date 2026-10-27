"""Folder-local collection counts; no target, quota or completion gate."""
import fcntl
import json
import os
from pathlib import Path
import time

SCHEMA = "manual_collection_v1"
WORKSPACE = Path(__file__).resolve().parents[1]


def validate_output(path):
    if not path:
        raise ValueError("Choose a Dataset Output folder first")
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not os.access(root,os.W_OK):
        raise ValueError("Choose an existing writable folder")
    # New sibling datasets under Square/raw are valid outputs. Protect the
    # original dataset itself, not the entire experiments workspace.
    experiments = (WORKSPACE/"experiments").resolve()
    square_raw = experiments/"mimicgen_square/datasets/raw"
    frozen_raw = square_raw/"restricted30_geomagic"
    if root == frozen_raw or frozen_raw in root.parents:
        raise ValueError("Existing fixed data is protected; choose a separate folder under datasets/raw")
    if root == experiments or root in experiments.parents:
        raise ValueError("Choose a dataset folder, not a shared workspace root")
    if experiments in root.parents and square_raw not in root.parents:
        raise ValueError("Choose a new folder under mimicgen_square/datasets/raw")
    summary=root/"collection_summary.json"
    if summary.exists() and json.loads(summary.read_text()).get("schema")!=SCHEMA:
        raise ValueError("collection_summary.json belongs to another application")
    return root


def counts(root):
    root=Path(root)
    success=failure=other=0
    saved_run_ids=[]
    # Existing recorder supports flat episodes and success/failure subfolders.
    paths=set(root.glob("*/metadata.json"))|set(root.glob("*/*/metadata.json"))
    for p in paths:
        if any(part.startswith('.') for part in p.relative_to(root).parts):continue
        try:
            m=json.loads(p.read_text())
            if m.get('task')!='mimicgen_square':continue
            saved_run_ids.append(m.get('collection_run_id'))
            if m.get('outcome')=='success':success+=1
            elif m.get('outcome')=='failure':failure+=1
            else:other+=1
        except (OSError,ValueError):pass
    runs={}
    log=root/'collection_runs.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                event=json.loads(line)
                if event.get('schema')!=SCHEMA:continue
                run=runs.setdefault(event['run_id'],{})
                run[event['event']]=event
            except (ValueError,KeyError):continue
    started=[v for v in runs.values() if 'started' in v]
    started_ids={key for key,value in runs.items() if 'started' in value}
    return dict(schema=SCHEMA,folder=str(root),success=success,failure=failure,
        other_saved=other,executions=len(started)+sum(run_id not in started_ids for run_id in saved_run_ids),
        active=sum('finished' not in r for r in started),limit=None)


def event(root,run_id,kind,**details):
    root=validate_output(root)
    with (root/'.collection_counts.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (root/'collection_runs.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(schema=SCHEMA,run_id=run_id,event=kind,
                timestamp=time.time(),**details),ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        value=counts(root)
        temporary=root/'.collection_summary.tmp'
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(root/'collection_summary.json')
    return value
