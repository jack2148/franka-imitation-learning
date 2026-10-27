#!/usr/bin/env python3
"""Validate a converted dataset through the installed LeRobot reader."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_lerobot_dataset(dataset_root: str | Path, raw_root: str | Path | None = None, *, write_report: bool = True, quick: bool = False) -> dict:
    # The HTTP backend may run in a restricted environment where the default
    # Hugging Face cache under the home directory is read-only.  LeRobot only
    # needs a local cache here; its dataset itself remains read-only.
    os.environ.setdefault("HF_HOME", "/tmp/mimic-hf-cache")
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root).expanduser().resolve()
    provenance = json.loads((root / "meta" / "mimic_conversion.json").read_text(encoding="utf-8"))
    camera_features = [
        key
        for key in provenance.get("observation_features", [])
        if key.startswith("observation.images.")
    ]
    if not camera_features:
        raise ValueError("conversion provenance does not declare image features")
    dataset = LeRobotDataset(repo_id=provenance["lerobot_repo_id"], root=root, return_uint8=True)
    errors: list[str] = []
    action_max = 0.0
    state_max = 0.0
    camera_ok = {key.removeprefix("observation.images."): True for key in camera_features}
    episodes = int(dataset.num_episodes)
    frames = int(len(dataset))
    if episodes != int(provenance["episodes"]): errors.append("episode count mismatch")
    if frames != int(provenance["frames"]): errors.append("frame count mismatch")
    episode_lengths = [int(record["frames"]) for record in provenance["source_episodes"]]
    if len(episode_lengths) != episodes or any(length <= 0 for length in episode_lengths) or sum(episode_lengths) != frames:
        errors.append("source episode lengths do not match LeRobot metadata")
    boundaries = [0]
    for length in episode_lengths:
        boundaries.append(boundaries[-1] + length)
    feature_shapes = {key: tuple(value["shape"]) for key, value in dataset.features.items()}
    if feature_shapes.get("action") != (8,): errors.append("action feature is not 8D")
    if feature_shapes.get("observation.state") != (16,): errors.append("state feature is not 16D")

    if quick:
        # Loading every high-resolution image is expensive for a folder
        # picker and training preflight.  Check both ends of every episode;
        # the LeRobot reader already checks the full table and metadata on load.
        indices = sorted({index for start, end in zip(boundaries, boundaries[1:]) for index in (start, end - 1)})
    else:
        indices = range(frames)
    for index in indices:
        item = dataset[index]
        episode = int(_numpy(item["episode_index"]).item())
        frame = int(_numpy(item["frame_index"]).item())
        timestamp = float(_numpy(item["timestamp"]).item())
        expected_episode = bisect_right(boundaries, index) - 1
        expected_frame = index - boundaries[expected_episode]
        if episode != expected_episode or frame != expected_frame or not np.isclose(timestamp, expected_frame / 20, atol=1e-4):
            errors.append(f"invalid in-episode sequence at global frame {index}")
            break
        for key in ("action", "observation.state"):
            if not np.isfinite(_numpy(item[key])).all(): errors.append(f"non-finite {key} at {index}")
        for key in camera_features:
            camera = key.removeprefix("observation.images.")
            camera_ok[camera] = camera_ok[camera] and _numpy(item[key]).shape == (3, 480, 640)

    if raw_root is not None:
        source = Path(raw_root).expanduser().resolve()
        global_index = 0
        for record in provenance["source_episodes"]:
            episode = source / record["directory"] if source.is_dir() and not (source / "metadata.json").is_file() else source
            with np.load(episode / "samples_20hz.npz", allow_pickle=False) as raw:
                for frame_index in sorted({0, int(record["frames"]) - 1}):
                    item = dataset[global_index + frame_index]
                    action_max = max(action_max, float(np.max(np.abs(_numpy(item["action"]) - raw["action"][frame_index]))))
                    state_max = max(state_max, float(np.max(np.abs(_numpy(item["observation.state"]) - raw["observation_state"][frame_index]))))
                    for camera in camera_ok:
                        converted = _numpy(item[f"observation.images.{camera}"])
                        if converted.shape == (3, 480, 640): converted = np.transpose(converted, (1, 2, 0))
                        if np.issubdtype(converted.dtype, np.floating) and converted.size and converted.max() <= 1.0:
                            converted = np.rint(converted * 255.0)
                        from PIL import Image
                        with Image.open(episode / "images" / camera / f"{frame_index:06d}.png") as image:
                            equal = np.array_equal(converted.astype(np.uint8), np.asarray(image.convert("RGB"), dtype=np.uint8))
                        camera_ok[camera] = camera_ok[camera] and equal
                global_index += int(record["frames"])
    result = {
        "valid": not errors and all(camera_ok.values()), "lerobot_load": True,
        "path": str(root), "type": "LEROBOT", "training_ready": not errors and all(camera_ok.values()),
        "episodes": episodes, "frames": frames, "samples": frames, "task": provenance["task"],
        "camera_schema": provenance.get("camera_schema"),
        "action_shape": feature_shapes.get("action"), "state_shape": feature_shapes.get("observation.state"),
        "action_max_abs_discrepancy": action_max, "state_max_abs_discrepancy": state_max,
        "operator_equivalent": camera_ok.get("operator", False),
        "front_equivalent": camera_ok.get("front", False),
        "wrist_equivalent": camera_ok.get("wrist", False),
        "camera_equivalent": camera_ok,
        "readiness": "READY_FOR_TRAINING" if not errors and all(camera_ok.values()) else "INVALID_LEROBOT_DATASET",
        "validation_scope": "episode_boundaries" if quick else "all_frames",
        "errors": errors,
    }
    if write_report:
        (root / "meta" / "mimic_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-root")
    args = parser.parse_args()
    result = validate_lerobot_dataset(args.dataset, args.raw_root)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
