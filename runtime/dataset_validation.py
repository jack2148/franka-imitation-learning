"""Read-only validation for Mimic raw episodes and LeRobot datasets."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from .camera_schema import (
    CURRENT_CAMERA_SCHEMA,
    camera_names_for_schema,
    detect_raw_camera_schema,
)
from .episode_naming import OUTCOME_DIRS


def _looks_like_episode(path: Path) -> bool:
    return path.is_dir() and ((path / "metadata.json").is_file() or (path / "samples_20hz.npz").is_file())


def _discover_episodes(root: Path) -> list[tuple[Path, str | None]]:
    """Find raw episodes under ``root``, both layouts side by side.

    Returns ``(episode_dir, expected_outcome)`` pairs, sorted by directory
    name. ``expected_outcome`` is ``"success"``/``"failure"`` for an episode
    found under ``root/success`` or ``root/failure`` (the current
    outcome-grouped layout), or ``None`` for a legacy flat episode found
    directly under ``root`` (no parent-folder outcome to cross-check).

    This looks only in the two known outcome subfolders plus root's direct
    children -- it does not recursively walk arbitrary depth, so it cannot
    accidentally ingest something from deeper inside an episode directory.
    """
    found: list[tuple[Path, str | None]] = []
    for child in root.iterdir():
        if child.is_dir() and child.name not in OUTCOME_DIRS and _looks_like_episode(child):
            found.append((child, None))
    for outcome in OUTCOME_DIRS:
        outcome_dir = root / outcome
        if not outcome_dir.is_dir():
            continue
        for child in outcome_dir.iterdir():
            if _looks_like_episode(child):
                found.append((child, outcome))
    return sorted(found, key=lambda pair: pair[0].name)


def validate_dataset(path: str | Path) -> dict:
    root = Path(path).expanduser().resolve()
    result = {"path": str(root), "valid": False, "training_ready": False,
              "type": "UNKNOWN", "episodes": 0, "samples": 0, "images": {}, "image_shapes": {},
              "task": None, "outcome": None, "duration_s": None, "camera_schema": None,
              "attempt_index": None, "recorded_at": None, "layout": None,
              "errors": [], "warnings": [], "checks": {},
              "required_fields": ["metadata.json", "initial_state.npz", "samples_20hz.npz"],
              "diagnostic_only_fields": ["original_1000hz.npz", "contact_replay_1000hz.npz", "contact_replay_metrics.json", "contact plots"]}
    if not root.is_dir():
        result["errors"].append("dataset directory does not exist")
        return result
    discovered = _discover_episodes(root)
    if discovered:
        children = []
        for episode_dir, expected_outcome in discovered:
            item = validate_dataset(episode_dir)
            item["layout"] = "CURRENT_OUTCOME_GROUPED" if expected_outcome is not None else "LEGACY_FLAT"
            if expected_outcome is not None:
                item["outcome_folder"] = expected_outcome
                matches = item.get("outcome") == expected_outcome
                item["checks"]["outcome_folder_matches_metadata"] = matches
                if not matches:
                    item["errors"].append(
                        f"parent folder outcome '{expected_outcome}' does not match "
                        f"metadata outcome '{item.get('outcome')}'"
                    )
                    item["valid"] = False
            children.append(item)
        names = [
            str(episode_dir.relative_to(root)) for episode_dir, _ in discovered
        ]
        tasks = {item.get("task") for item in children if item.get("task")}
        schemas = {item.get("camera_schema") for item in children if item.get("camera_schema")}
        result.update(type="RAW_MIMIC", episodes=len(children), samples=sum(item["samples"] for item in children), task=next(iter(tasks)) if len(tasks) == 1 else "mixed")
        result["camera_schema"] = next(iter(schemas)) if len(schemas) == 1 else "mixed"
        result["outcomes"] = {name: sum(item.get("outcome") == name for item in children) for name in ("success", "failure", "diagnostic")}
        result["valid"] = all(item["valid"] for item in children)
        result["training_ready"] = False
        result["readiness"] = "REQUIRES_PROCESSING_TO_LEROBOT" if result["valid"] else "INVALID_RAW_DATASET"
        result["errors"] = [f"{names[index]}: {error}" for index, item in enumerate(children) for error in item["errors"]]
        result["images"] = {
            camera: sum(item["images"].get(camera, 0) for item in children)
            for camera in ("operator", "front", "wrist")
        }
        return result
    info = root / "meta" / "info.json"
    if info.is_file():
        result["type"] = "LEROBOT"
        try:
            from dataset.validate_lerobot_dataset import validate_lerobot_dataset
            checked = validate_lerobot_dataset(root, write_report=False, quick=True)
            result.update(checked)
            result["images"] = {
                "operator": checked["frames"] if checked.get("operator_equivalent") else 0,
                "front": checked["frames"] if checked.get("front_equivalent") else 0,
                "wrist": checked["frames"] if checked["wrist_equivalent"] else 0,
            }
            result["checks"].update({"meta_info": True, "official_lerobot_load": checked["lerobot_load"], "action_shape": checked["action_shape"] == (8,), "state_shape": checked["state_shape"] == (16,)})
        except Exception as exc:
            result["errors"].append(f"official LeRobot validation failed: {exc}")
        return result
    result["type"] = "RAW"
    metadata_path = root / "metadata.json"
    sample_path = root / "samples_20hz.npz"
    if not metadata_path.is_file() or not sample_path.is_file():
        result["errors"].append("missing metadata.json or samples_20hz.npz")
        return result
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result["task"] = metadata.get("task")
        result["outcome"] = metadata.get("outcome")
        result["attempt_index"] = metadata.get("attempt_index")
        result["recorded_at"] = metadata.get("recorded_at")
        result["episodes"] = 1
        result["checks"]["metadata"] = bool(metadata.get("task") and metadata.get("outcome"))
        if not result["checks"]["metadata"]:
            result["errors"].append("metadata must contain task and outcome")
        if metadata.get("task") not in {
            "fr3-pick-place",
            "pick_and_place",
            "mimicgen_square",
        }:
            result["errors"].append("unexpected task")
        with np.load(sample_path, allow_pickle=False) as data:
            required = {"sim_time", "control_step", "dataset_frame_index", "observation_state", "arm_qpos", "arm_qvel", "gripper_qpos", "action", "ee_pose", "object_pose", "goal_position", "success"}
            missing = sorted(required - set(data.files))
            if missing:
                result["errors"].append("missing fields: " + ", ".join(missing))
            n = int(data["action"].shape[0]) if "action" in data and data["action"].ndim == 2 else 0
            result["samples"] = n
            from .sample_weight import validate_weight, validate_weight_csv
            labels = validate_weight(data, n, metadata)
            if labels is not None:
                if metadata.get("sample_weight"):
                    validate_weight_csv(root, data)
                result["checks"]["sample_weight"] = True
            result["checks"]["action_shape"] = "action" in data and tuple(data["action"].shape[1:]) == (8,)
            if not result["checks"]["action_shape"]:
                result["errors"].append("action must have shape (N, 8)")
            finite = True
            for name in data.files:
                value = data[name]
                if value.shape and value.shape[0] != n:
                    result["errors"].append(f"sample count mismatch in {name}")
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    finite = False
                    result["errors"].append(f"non-finite values in {name}")
            result["checks"]["finite"] = finite
            if n and "sim_time" in data:
                dt = np.diff(data["sim_time"])
                result["duration_s"] = float(data["sim_time"][-1] - data["sim_time"][0])
                result["checks"]["timestamps"] = bool(np.all(dt > 0) and np.allclose(dt, 0.05, atol=0.006))
                if not result["checks"]["timestamps"]: result["errors"].append("dataset timestamps are not approximately 20 Hz")
            if n and "control_step" in data:
                ds = np.diff(data["control_step"])
                result["checks"]["control_alignment"] = bool(np.all(ds == 50))
                if not result["checks"]["control_alignment"]: result["warnings"].append("control-step spacing is not exactly 50")
            result["checks"]["temporal_convention"] = metadata.get("temporal_convention") in {None, "observation_t_before_step -> action_t"}
            if not result["checks"]["temporal_convention"]:
                result["errors"].append("unexpected observation/action temporal convention")
            if n and "dataset_frame_index" in data:
                result["checks"]["frame_indices"] = bool(np.array_equal(data["dataset_frame_index"], np.arange(n)))
                if not result["checks"]["frame_indices"]: result["errors"].append("dataset frame indices are not contiguous")
        camera_schema = detect_raw_camera_schema(root, metadata)
        cameras = camera_names_for_schema(camera_schema)
        result["camera_schema"] = camera_schema
        result["required_fields"].extend(f"images/{camera}" for camera in cameras)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        result["errors"].append(f"unable to read raw episode: {exc}")
    if not (root / "initial_state.npz").is_file():
        result["errors"].append("missing initial_state.npz")
    cameras = camera_names_for_schema(result["camera_schema"]) if result["camera_schema"] else ()
    for camera in cameras:
        directory = root / "images" / camera
        files = sorted(directory.glob("*.png")) if directory.is_dir() else []
        result["images"][camera] = len(files)
        expected = [directory / f"{i:06d}.png" for i in range(result["samples"])]
        ok = len(files) == result["samples"] and files == expected
        shape = None
        if files:
            try:
                for file in files:
                    with Image.open(file) as image:
                        current_shape = [image.height, image.width, 3 if image.mode == "RGB" else 0]
                        shape = shape or current_shape
                        ok = ok and current_shape == shape and image.mode == "RGB" and current_shape == [480, 640, 3]
            except OSError:
                ok = False
        result["image_shapes"][camera] = shape
        result["checks"][f"images_{camera}"] = ok
        if not ok: result["errors"].append(f"{camera} image sequence/count is invalid")
    if result["camera_schema"] == CURRENT_CAMERA_SCHEMA:
        # Every display-only view, current and retired. Only operator and
        # wrist may ever appear in an operator_wrist_v1 episode.
        ui_only_cameras = ("front", "front_operator", "left_side", "top_down")
        for ui_only in ui_only_cameras:
            stored = root / "images" / ui_only
            if stored.is_dir() and any(stored.glob("*.png")):
                result["errors"].append(f"UI-only camera {ui_only} must not be stored")
        result["checks"]["ui_only_not_stored"] = not any(
            (root / "images" / name).is_dir()
            and any((root / "images" / name).glob("*.png"))
            for name in ui_only_cameras
        )
    result["valid"] = not result["errors"]
    result["training_ready"] = False
    result["readiness"] = "REQUIRES_PROCESSING_TO_LEROBOT" if result["valid"] else "INVALID_RAW_DATASET"
    if result["valid"]:
        result["warnings"].append("raw Mimic episode is not directly consumable by the existing ACT/DP entrypoints")
    return result
