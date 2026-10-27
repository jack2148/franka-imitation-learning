"""Success-gated collection plans, independent of WORK UI task branches."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time
from typing import Any

from environments.mimicgen_square.config import SQUARE_RESTRICTED30_V1


@dataclass(frozen=True)
class CollectionPlan:
    distribution_id: str
    title: str
    label: str
    task: str
    seed_start: int
    seed_end: int
    advance_condition: str
    world_xy_bounds: tuple[tuple[float, float], tuple[float, float]]
    yaw_center: float
    yaw_half_range: float
    # Relative to the experiments/ workspace; keeps each version's recordings apart.
    default_raw_root: str

    @property
    def target(self) -> int:
        return self.seed_end - self.seed_start + 1

    def accepted_seeds(self, raw_root: Path) -> set[int]:
        """Count only validated, finalized production successes for this plan."""
        accepted: set[int] = set()
        success_dir = raw_root / "success"
        if not success_dir.is_dir():
            return accepted
        for path in success_dir.glob("*/metadata.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                seed = metadata.get("distribution_seed")
                if (
                    metadata.get("task") == self.task
                    and metadata.get("initial_state_distribution") == self.distribution_id
                    and metadata.get("outcome") == "success"
                    and metadata.get("collection_accepted") is True
                    and isinstance(metadata.get("attempt_index"), int)
                    and isinstance(seed, int)
                    and self.seed_start <= seed <= self.seed_end
                ):
                    accepted.add(seed)
            except (OSError, ValueError, TypeError):
                continue
        return accepted

    def ensure_isolated_root(self, raw_root: Path) -> None:
        """Refuse to mix this version with prior grouped production recordings."""
        if raw_root.name in {"raw", "uniform50"}:
            raise ValueError(
                f"{self.label} requires a separate raw root, not {raw_root.name}"
            )
        for outcome in ("success", "failure"):
            for path in (raw_root / outcome).glob("*/metadata.json"):
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    continue
                if metadata.get("initial_state_distribution") != self.distribution_id:
                    raise ValueError(
                        f"raw root contains another distribution: {path}"
                    )

    def next_seed(self, raw_root: Path) -> int | None:
        accepted = self.accepted_seeds(raw_root)
        return next((seed for seed in range(self.seed_start, self.seed_end + 1)
                     if seed not in accepted), None)

    def progress(self, raw_root: Path, *, last_outcome: str | None = None) -> dict[str, Any]:
        """UI summary. ``last_outcome`` is the just-finished attempt, or
        ``"success_not_saved"`` when a SUCCESS failed to save/validate."""
        accepted = self.accepted_seeds(raw_root)
        seed = next((s for s in range(self.seed_start, self.seed_end + 1)
                     if s not in accepted), None)
        status = "COLLECTION COMPLETE" if seed is None else (
            "RETRY SAME INITIAL STATE" if last_outcome == "failure" else
            "SUCCESS NOT SAVED · RETRY SAME INITIAL STATE"
            if last_outcome == "success_not_saved" else
            "SUCCESS" if last_outcome == "success" else "READY"
        )
        return {
            "distribution_id": self.distribution_id, "title": self.title,
            "label": self.label,
            "current_seed": seed, "completed": len(accepted), "target": self.target,
            "remaining": self.target - len(accepted), "complete": seed is None,
            "status": status, "last_outcome": last_outcome,
            "seed_start": self.seed_start, "seed_end": self.seed_end,
            "advance_condition": self.advance_condition,
            "world_xy_bounds": self.world_xy_bounds,
            "yaw_center": self.yaw_center, "yaw_half_range": self.yaw_half_range,
        }


SQUARE_RESTRICTED30_PLAN = CollectionPlan(
    distribution_id=SQUARE_RESTRICTED30_V1.distribution_id,
    title="Square Collection",
    label="Restricted 30 v1", task="mimicgen_square",
    seed_start=100, seed_end=129, advance_condition="SUCCESS_ONLY",
    world_xy_bounds=SQUARE_RESTRICTED30_V1.world_xy_bounds(),
    yaw_center=SQUARE_RESTRICTED30_V1.yaw_center,
    yaw_half_range=SQUARE_RESTRICTED30_V1.yaw_half_range,
    default_raw_root="mimicgen_square/datasets/raw/restricted30",
)

COLLECTION_PLANS = {SQUARE_RESTRICTED30_PLAN.distribution_id: SQUARE_RESTRICTED30_PLAN}


def collection_plan(distribution_id: str | None) -> CollectionPlan | None:
    if distribution_id is None:
        return None
    try:
        return COLLECTION_PLANS[distribution_id]
    except KeyError as error:
        if distribution_id.startswith('square_geomagic_xy50_v'):
            from .stratified_plan import load_manifest
            return XYCollectionPlan.from_document(load_manifest(distribution_id))
        raise ValueError(f"unknown collection distribution: {distribution_id}") from error


def accept_finalized_success(
    episode_dir: Path, plan: CollectionPlan, validation: dict[str, Any],
) -> None:
    """Mark an already-saved, validated success as one accepted seed.

    The lock prevents two WORK processes from accepting duplicate seeds during
    concurrent restarts. A failed validation/save never writes the marker.
    """
    if not validation.get("valid"):
        raise RuntimeError("saved SUCCESS episode failed dataset validation; seed not advanced")
    if episode_dir.parent.name != "success":
        raise ValueError("accepted episode must be under raw/success")
    root = episode_dir.parent.parent
    metadata_path = episode_dir / "metadata.json"
    with (root / ".collection-plan.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(plan, XYCollectionPlan):
            plan.require_frozen()
            plan.validate_episode_metadata(metadata)
        seed = metadata.get("distribution_seed")
        if (
            metadata.get("task") != plan.task
            or metadata.get("initial_state_distribution") != plan.distribution_id
            or metadata.get("outcome") != "success"
            or not isinstance(seed, int)
            or not plan.seed_start <= seed <= plan.seed_end
            or not isinstance(metadata.get("attempt_index"), int)
        ):
            raise ValueError("saved success metadata does not match active collection plan")
        if seed in plan.accepted_seeds(root):
            raise RuntimeError(f"seed {seed} already has an accepted success")
        if isinstance(plan, XYCollectionPlan) and seed != plan.next_seed(root):
            raise ValueError('XY50 success must match the first unaccepted collection order')
        metadata["collection_accepted"] = True
        metadata["collection_accepted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        temporary = metadata_path.with_name(".metadata.accepted.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)


@dataclass(frozen=True)
class XYCollectionPlan(CollectionPlan):
    """Immutable XY intent; accepted status stays in finalized episode metadata."""
    document: dict
    directory: Path

    @classmethod
    def from_document(cls, document, directory=None):
        from .stratified_plan import PLAN_DIRECTORY
        if 'training_plan' in document['configuration']:
            raise ValueError('evaluation states cannot be used as demonstration collection plans')
        w=document['workspace']
        yaw=document['object_yaw_world_rad'] - 3.141592653589793/2
        return cls(document['plan_name'],'Geomagic XY Collection',document['plan_name'],
            'mimicgen_square',0,len(document['states'])-1,'SUCCESS_ONLY',
            ((w['x_min'],w['x_max']),(w['y_min'],w['y_max'])),yaw,0.,
            document['configuration']['default_raw_root'],deepcopy(document),
            Path(directory or PLAN_DIRECTORY))

    def require_frozen(self):
        from .stratified_plan import load_manifest
        current=load_manifest(self.distribution_id,self.directory,require_frozen=True)
        if current['content_sha256'] != self.document['content_sha256']:
            raise ValueError('active plan changed after it was loaded')

    def planned_state(self, order):
        if type(order) is not int or not self.seed_start <= order <= self.seed_end:
            raise ValueError('collection order is outside the plan')
        return deepcopy(next(s for s in self.document['states'] if s['collection_order']==order))

    def yaw_offset(self, canonical_yaw):
        from environments.mimicgen_square.config import wrap_to_pi
        return wrap_to_pi(canonical_yaw-self.yaw_center)

    def episode_fields(self, order, *, attempt_number):
        state=self.planned_state(order)
        return dict(collection_plan=self.distribution_id,
            planned_state_id=state['planned_state_id'], planned_state_index=state['plan_index'],
            grid_rows=5, grid_cols=5, grid_cell_row=state['cell_row'],
            grid_cell_col=state['cell_col'], grid_cell_id=state['cell_id'],
            planned_object_xy=[state['x'],state['y']], collection_order=order,
            collection_rng_seed=self.document['seed'],
            collection_shuffle_seed=self.document['shuffle_seed'],
            collection_plan_sha256=self.document['content_sha256'],
            attempt_number=attempt_number)

    def validate_episode_metadata(self, metadata):
        order=metadata.get('distribution_seed')
        expected=self.episode_fields(order,attempt_number=metadata.get('attempt_number'))
        if any(metadata.get(key)!=value for key,value in expected.items()):
            raise ValueError('saved metadata does not match planned state/cell/XY/order/checksum')
        if (metadata.get('collection_plan_id')!=self.distribution_id
                or metadata.get('initial_state_distribution')!=self.distribution_id
                or metadata.get('task')!=self.task
                or metadata.get('hardware_source')!='geomagic'
                or metadata.get('input_source')!='physical_hdapi'
                or metadata.get('synthetic_input') is not False
                or metadata.get('hardware_verified_by_this_run') is not True):
            raise ValueError('XY50 acceptance requires verified physical Geomagic provenance')
        if type(metadata.get('attempt_number')) is not int or metadata['attempt_number'] < 1:
            raise ValueError('attempt_number must be positive')
        pose=metadata.get('initial_object_pose',[])
        import math
        yaw=self.document['object_yaw_world_rad']
        fixed=[*expected['planned_object_xy'],self.document['object_z'],
               math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
        if len(pose)!=7 or any(abs(a-b)>1e-12 or not math.isfinite(a) for a,b in zip(pose,fixed)):
            raise ValueError('initial object pose is not the exact planned XY/fixed Z/orientation')

    def accepted_seeds(self, raw_root):
        accepted=set()
        for path in (Path(raw_root)/'success').glob('*/metadata.json'):
            try:
                metadata=json.loads(path.read_text())
                self.validate_episode_metadata(metadata)
                if (metadata.get('outcome')=='success' and metadata.get('collection_accepted') is True
                        and type(metadata.get('attempt_index')) is int):
                    accepted.add(metadata['distribution_seed'])
            except (OSError,ValueError,TypeError,KeyError):
                continue
        return accepted

    def failed_states(self, raw_root):
        failed=set()
        for path in (Path(raw_root)/'failure').glob('*/metadata.json'):
            try:
                metadata=json.loads(path.read_text())
                self.validate_episode_metadata(metadata)
                if metadata.get('outcome')=='failure':failed.add(metadata['distribution_seed'])
            except (OSError,ValueError,TypeError,KeyError):continue
        ledger = self._attempt_ledger(Path(raw_root))
        failed.update(s['collection_order'] for s in self.document['states']
            if ledger['attempts'].get(s['planned_state_id'], 0) > 1)
        for path in (Path(raw_root)/'success').glob('*/metadata.json'):
            try:
                metadata=json.loads(path.read_text())
                self.validate_episode_metadata(metadata)
                if metadata['attempt_number'] > 1: failed.add(metadata['distribution_seed'])
            except (OSError,ValueError,TypeError,KeyError):continue
        return failed

    def attempt_number(self, raw_root, order):
        attempts=[]
        # Finalized outcomes and diagnostics all retain state identity. A crash
        # without finalized metadata does not consume an accepted state.
        for path in Path(raw_root).glob('*/metadata.json'):
            attempts.append(path)
        for outcome in ('success','failure'):
            attempts.extend((Path(raw_root)/outcome).glob('*/metadata.json'))
        highest=0
        state_id=self.planned_state(order)['planned_state_id']
        for path in attempts:
            try:
                metadata=json.loads(path.read_text())
                attempt=metadata.get('attempt_number')
                if (metadata.get('collection_plan_sha256')==self.document['content_sha256']
                        and metadata.get('planned_state_id')==state_id
                        and type(attempt) is int):highest=max(highest,attempt)
            except (OSError,ValueError,TypeError):continue
        ledger = self._attempt_ledger(Path(raw_root))
        return max(highest, ledger['attempts'].get(state_id, 0))+1

    def _attempt_ledger(self, raw_root):
        path = raw_root / '.xy50-attempts.json'
        if not path.exists():
            return dict(collection_plan=self.distribution_id,
                        collection_plan_sha256=self.document['content_sha256'], attempts={})
        ledger = json.loads(path.read_text())
        if (not isinstance(ledger, dict) or ledger.get('collection_plan') != self.distribution_id
                or ledger.get('collection_plan_sha256') != self.document['content_sha256']
                or not isinstance(ledger.get('attempts'), dict)
                or any(type(v) is not int or v < 1 for v in ledger['attempts'].values())):
            raise ValueError('attempt accounting does not match frozen plan; cannot silently reset it')
        return ledger

    def reserve_attempt(self, raw_root, order):
        """Persist attempt identity even for discard/crash; never advance XY."""
        from .stratified_plan import write_json
        self.require_frozen()
        raw_root = Path(raw_root)
        self.ensure_isolated_root(raw_root)
        raw_root.mkdir(parents=True, exist_ok=True)
        with (raw_root / '.collection-plan.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if order != self.next_seed(raw_root):
                raise ValueError('attempt must use the first unaccepted collection order')
            number = self.attempt_number(raw_root, order)
            ledger = self._attempt_ledger(raw_root)
            ledger['attempts'][self.planned_state(order)['planned_state_id']] = number
            write_json(raw_root / '.xy50-attempts.json', ledger)
            return number

    def progress(self, raw_root, *, last_outcome=None):
        result=super().progress(raw_root,last_outcome=last_outcome)
        state=self.planned_state(result['current_seed']) if result['current_seed'] is not None else None
        result.update(plan_status=self.document['status'],planned_state=state,
            attempt_number=self.attempt_number(raw_root,result['current_seed']) if state else None,
            collection_rng_seed=self.document['seed'])
        if self.document['status']=='DRAFT':result['status']='DRAFT · VALIDATE WORKSPACE AND FREEZE'
        elif last_outcome in ('discard','abort','reset'):result['status']='RETRY SAME INITIAL STATE'
        return result
