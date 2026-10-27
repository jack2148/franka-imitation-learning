"""Read-only ANALYZE access to normalized evaluation artifacts."""

import threading
from ..contracts import AdapterStatus
from ..paths import shared_runs_root
from .metrics import enrich
from .parser import parse_run
from .pairing import compatibility, pair
from .run_index import RunIndex


class AnalyzeAdapter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index = RunIndex(shared_runs_root() / "evaluation")
        self._parsed: dict[str, tuple[float, dict]] = {}

    def status(self) -> AdapterStatus:
        return AdapterStatus("analyze", self._index.root.is_dir(), "Evaluation artifact reader is ready; analysis is read-only.")

    def snapshot(self) -> dict:
        with self._lock:
            paths = self._index.paths()
            runs, details = [], {}
            for path in paths:
                mtime = max((item.stat().st_mtime for item in path.glob("*.json") if item.is_file()), default=path.stat().st_mtime)
                key = str(path)
                cached = self._parsed.get(key)
                if cached is None or cached[0] != mtime:
                    self._parsed[key] = (mtime, enrich(parse_run(path)))
                run = self._parsed[key][1]
                details[run["run_id"]] = run
                runs.append({key: run.get(key) for key in ("run_id", "path", "policy", "checkpoint", "dataset", "seed", "episodes", "success_count", "failure_count", "success_rate", "max_steps")})
            return {
                "data_available": bool(runs),
                "runs": runs,
                "details": details,
                "region": {"status": "AVAILABLE" if any(run.get("region_summary", {}).get("available") for run in details.values()) else "NOT AVAILABLE", "version": "fr3_push_regions_v1" if any(run.get("region_summary", {}).get("available") for run in details.values()) else None, "message": "Policy success by fixed Region v1 initial-state groups."},
                "next_collection": {"status": "NOT COMPUTED", "message": "Requires region-conditioned evaluation, coverage statistics, and feasibility estimates."},
            }
