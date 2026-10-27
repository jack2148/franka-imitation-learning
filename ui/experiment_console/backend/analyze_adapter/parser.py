"""Normalize the existing evaluation JSON/CSV artifacts."""

from pathlib import Path
import csv
import json
import math
from statistics import median

from .failure_analysis import classify_episode


def _region_summary(episodes: list[dict], config: dict) -> dict:
    version = config.get("region_version")
    # Legacy runs intentionally remain unclassified even when their pose can
    # be inferred from old trajectory CSVs.
    if not version:
        return {"available": False, "version": None, "regions": []}
    rows = []
    for label in ("A", "B", "C"):
        subset = [item for item in episodes if item.get("region") == label]
        successes = sum(item.get("success") is True for item in subset)
        rows.append({"region": label, "episodes": len(subset), "success_count": successes, "success_rate": successes / len(subset) if subset else None})
    return {"available": True, "version": version, "regions": rows}


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _array(row: dict, prefix: str):
    direct = row.get(prefix)
    if isinstance(direct, list):
        try:
            return [float(value) for value in direct]
        except (TypeError, ValueError):
            return None
    values = []
    for index in range(7):
        value = _float(row.get(f"{prefix}_{index}"))
        if value is None:
            return None
        values.append(value)
    return values


def _training_lineage(checkpoint: str | None) -> dict:
    if not checkpoint:
        return {}
    path = Path(checkpoint).expanduser()
    train_config = path / "train_config.json"
    data = _json(train_config) if train_config.is_file() else {}
    policy = data.get("policy") if isinstance(data.get("policy"), dict) else {}
    optimizer = data.get("optimizer") if isinstance(data.get("optimizer"), dict) else {}
    return {
        "seed": data.get("seed"), "steps": data.get("steps"),
        "batch_size": data.get("batch_size"), "learning_rate": optimizer.get("lr"),
        "checkpoint_step": path.parent.name if path.name == "pretrained_model" else None,
        "policy_config": policy,
        "available": bool(data),
    }


def parse_run(path: Path) -> dict:
    config = _json(path / "evaluation_config.json")
    summary = _json(path / "evaluation_summary.json")
    rows_by_episode: dict[int, list[dict]] = {}
    jsonl_path = path / "debug_steps.jsonl"
    if not jsonl_path.is_file():
        csv_path = path / "debug_steps.csv"
        try:
            with csv_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    try:
                        episode = int(row.get("episode_id", row.get("episode", 0)))
                    except (TypeError, ValueError):
                        continue
                    rows_by_episode.setdefault(episode, []).append(row)
        except OSError:
            pass
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode = int(row.get("episode_id", row.get("episode", 0)))
            rows_by_episode.setdefault(episode, []).append(row)
    except (OSError, ValueError, TypeError):
        pass

    records = summary.get("records") if isinstance(summary.get("records"), list) else []
    first_record = records[0] if records and isinstance(records[0], dict) else {}
    episodes = []
    count = max(len(records), len(rows_by_episode))
    for index in range(count):
        record = records[index] if index < len(records) and isinstance(records[index], dict) else {}
        episode_id = int(record.get("episode_index", index))
        rows = rows_by_episode.get(episode_id, [])
        object_path = []
        final_distance = None
        final_ee = None
        if rows:
            for row in rows:
                x = _float(row.get("object_x", row.get("block_x")))
                y = _float(row.get("object_y", row.get("block_y")))
                if x is not None and y is not None:
                    object_path.append({"x": x, "y": y})
            final_distance = _float(rows[-1].get("goal_distance"))
        ee_rows = [{"x": _float(row.get("ee_x")), "y": _float(row.get("ee_y")), "z": _float(row.get("ee_z"))} for row in rows if _float(row.get("ee_x")) is not None]
        action_rows = [_array(row, "predicted_action") for row in rows]
        executed_rows = [_array(row, "executed_action") for row in rows]
        steps = record.get("steps", len(rows) or None)
        if len(object_path) > 600:
            stride = max(1, len(object_path) // 600)
            object_path = object_path[::stride]
        episode = {
            "episode_id": episode_id,
            "eval_seed": record.get("seed"),
            "success": record.get("success"),
            "termination_reason": record.get("termination_reason"),
            "steps": steps,
            "duration": record.get("duration"),
            "initial_ee_pose": record.get("initial_ee_pose"),
            "initial_object_pose": record.get("initial_object_pose") or (object_path[0] if object_path else None),
            "region": record.get("region"),
            "region_version": record.get("region_version"),
            "evaluation_state_id": record.get("evaluation_state_id"),
            "goal_pose": ({"x": _float(rows[0].get("goal_x")), "y": _float(rows[0].get("goal_y"))} if rows else None),
            "final_ee_pose": record.get("final_ee_pose") or final_ee,
            "final_object_pose": record.get("final_object_pose") or (object_path[-1] if object_path else None),
            "final_goal_distance": final_distance,
            "ee_trajectory": ee_rows or None,
            "object_trajectory": object_path or None,
            "predicted_actions": action_rows if action_rows and all(item is not None for item in action_rows) else None,
            "executed_actions": executed_rows if executed_rows and all(item is not None for item in executed_rows) else None,
        }
        if episode["predicted_actions"] and episode["executed_actions"]:
            deltas = [math.sqrt(sum((a - b) ** 2 for a, b in zip(predicted, executed))) for predicted, executed in zip(episode["predicted_actions"], episode["executed_actions"])]
            episode["action_execution_delta_mean"] = sum(deltas) / len(deltas) if deltas else None
            episode["action_execution_delta_max"] = max(deltas) if deltas else None
        else:
            episode["action_execution_delta_mean"] = None
            episode["action_execution_delta_max"] = None
        episode["failure_type"] = classify_episode(episode) if episode.get("success") is not True else None
        episodes.append(episode)

    success_count = sum(episode.get("success") is True for episode in episodes)
    failed = len(episodes) - success_count
    distances = [episode["final_goal_distance"] for episode in episodes if episode.get("final_goal_distance") is not None]
    steps = [float(episode["steps"]) for episode in episodes if episode.get("steps") is not None]
    return {
        "run_id": path.name,
        "path": str(path),
        "policy": config.get("policy") or first_record.get("policy"),
        "checkpoint": config.get("checkpoint") or first_record.get("checkpoint"),
        "dataset": config.get("dataset") or config.get("dataset_root") or first_record.get("dataset"),
        "seed": config.get("seed"),
        "episodes": len(episodes),
        "success_count": success_count,
        "failure_count": failed,
        "success_rate": (success_count / len(episodes) if episodes else None),
        "max_steps": config.get("max_steps"),
        "success_criterion": config.get("success_criterion"),
        "region": config.get("region"),
        "region_version": config.get("region_version"),
        "evaluation_state_set": config.get("evaluation_state_set"),
        "region_summary": _region_summary(episodes, config),
        "config": config,
        "training": _training_lineage(config.get("checkpoint")),
        "episode_results": episodes,
        "summary": {
            "mean_steps": sum(steps) / len(steps) if steps else None,
            "median_steps": median(steps) if steps else None,
            "mean_final_goal_distance": sum(distances) / len(distances) if distances else None,
            "median_final_goal_distance": median(distances) if distances else None,
            "mean_ee_trajectory_length": None,
        },
    }
