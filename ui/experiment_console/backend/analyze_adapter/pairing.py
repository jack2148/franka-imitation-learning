"""Compatibility and conservative pairing for baseline comparisons."""


def compatibility(a: dict, b: dict) -> dict:
    warnings = []
    for label in ("dataset", "max_steps", "success_criterion"):
        left, right = a.get(label), b.get(label)
        if left is not None and right is not None and left != right:
            warnings.append(f"{label} differs: {left} vs {right}")
    compatible = a.get("dataset") == b.get("dataset") and not warnings
    return {"compatible": compatible, "warnings": warnings}


def pair(a: dict, b: dict) -> list[dict]:
    right = {episode.get("eval_seed"): episode for episode in b.get("episode_results", []) if episode.get("eval_seed") is not None}
    pairs = []
    for left in a.get("episode_results", []):
        match = right.get(left.get("eval_seed"))
        if match is None:
            continue
        left_success, right_success = left.get("success") is True, match.get("success") is True
        category = "Both Success" if left_success and right_success else "Both Failure" if not left_success and not right_success else "ACT Success / DP Failure" if left_success else "ACT Failure / DP Success"
        pairs.append({"eval_seed": left.get("eval_seed"), "act": left, "dp": match, "category": category})
    return pairs
