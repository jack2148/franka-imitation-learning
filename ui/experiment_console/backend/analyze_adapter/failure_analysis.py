"""Conservative, observable rollout failure labels."""


def classify_episode(episode: dict) -> str:
    reason = str(episode.get("termination_reason") or "").lower()
    if reason in {"max_steps", "timeout", "timelimit"} or "timeout" in reason:
        return "TIMEOUT"
    # Existing artifacts do not contain contact state or a reliable crossing
    # signal. Do not infer these categories from motion alone.
    return "UNCLASSIFIED"
