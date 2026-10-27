"""Derived baseline metrics from normalized evaluation runs."""


def trajectory_length(points: list[dict] | None) -> float | None:
    if not points or len(points) < 2:
        return None
    return sum(((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5 for a, b in zip(points, points[1:]))


def failure_counts(run: dict) -> list[dict]:
    labels = ["TIMEOUT", "NO_CONTACT", "CONTACT_NO_PROGRESS", "OVERSHOOT", "WRONG_DIRECTION", "UNCLASSIFIED"]
    counts = {label: 0 for label in labels}
    for episode in run.get("episode_results", []):
        if episode.get("success") is not True:
            counts[episode.get("failure_type") or "UNCLASSIFIED"] += 1
    return [{"label": label, "count": counts[label]} for label in labels]


def enrich(run: dict) -> dict:
    summary = dict(run.get("summary", {}))
    lengths = [trajectory_length(e.get("ee_trajectory")) for e in run.get("episode_results", [])]
    lengths = [value for value in lengths if value is not None]
    summary["mean_ee_trajectory_length"] = sum(lengths) / len(lengths) if lengths else None
    return {**run, "summary": summary, "failure_counts": failure_counts(run)}
