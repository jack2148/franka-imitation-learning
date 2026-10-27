"""Manual collection using read-only XY50 TRAIN initial coordinates.

No frozen ledger or prior dataset is reused as an output. The coordinate list repeats after its last entry; it imposes no collection limit.
"""
import json
import os
from pathlib import Path
import time

TRAIN_DISTRIBUTION = "square_geomagic_xy50_v1"


def source_training_plan():
    from .stratified_plan import digest, fixed_context
    # Evidence is read-only and separate from the live collection-plan registry.
    # The original seal check and coordinate-only provenance remain unchanged.
    coordinate_directory = Path(__file__).resolve().parents[1] / "evidence" / "collection"
    path=coordinate_directory/(TRAIN_DISTRIBUTION+".plan.json")
    document=json.loads(path.read_text())
    seal=json.loads((coordinate_directory/(TRAIN_DISTRIBUTION+".frozen.json")).read_text())
    content={k:v for k,v in document.items() if k not in {"status","content_sha256"}}
    if (document.get("status")!="FROZEN" or digest(content)!=document.get("content_sha256")
        or seal.get(TRAIN_DISTRIBUTION)!=document["content_sha256"] or len(document["states"])!=50):
        raise ValueError("Source TRAIN coordinates failed their original seal verification")
    return document,dict(source=str(path),source_sha256=document["content_sha256"],
        source_kind="TRAIN initial coordinates only",current_fixed_context=fixed_context(),
        original_fixed_context=document["fixed_context"],
        original_environment_equivalence_claimed=False,canonical_plan_acceptance=False)


def apply_research_initial_state(runner, index):
    """Set only the new research runner's object pose, never the canonical plan."""
    import numpy as np
    import mujoco
    document,provenance=source_training_plan()
    selected=next(s for s in document["states"] if s["collection_order"]==index % len(document["states"]))
    yaw=document["object_yaw_world_rad"]
    address=runner.object_qpos_address
    runner.data.qpos[address:address+7]=[selected["x"],selected["y"],document["object_z"],
                                      np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
    dof=runner.env._object_dof_address
    runner.data.qvel[dof:dof+6]=0.
    runner.env._last_reset_yaw=yaw
    runner.env._last_canonical_yaw=yaw-np.pi/2
    mujoco.mj_forward(runner.model,runner.data)
    runner.backend.reset_controller_state()
    runner._initial_state=runner.capture_physics()
    return dict(provenance,selected_state=selected,coordinate_cycle=index // len(document["states"]))


def enabled():
    return os.environ.get("MIMIC_WEIGHT_COLLECTION") == "1"

