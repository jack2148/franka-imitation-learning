"""Read-only CSV/JSONL helpers for console analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import threading
import time

from ..contracts import TrajectorySummary


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def summarize_trajectory(path: Path) -> TrajectorySummary:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader, []))
            rows = sum(1 for _ in reader)
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        header = tuple(records[0].keys()) if records else ()
        rows = len(records)
    else:
        raise ValueError("trajectory_reader supports .csv and .jsonl only")
    return TrajectorySummary(str(path), rows, header)


def read_xy_trajectory(
    path: Path,
    *,
    x_fields: tuple[str, ...],
    y_fields: tuple[str, ...],
    limit: int = 600,
) -> dict:
    """Read a real CSV trajectory into a small JSON-safe representation."""
    path = Path(path).expanduser()
    if not path.is_file():
        return {"available": False, "source": str(path), "points": [], "message": "source unavailable"}
    if path.suffix.lower() != ".csv":
        return {"available": False, "source": str(path), "points": [], "message": "CSV source required"}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    x_name = next((name for name in x_fields if name in (reader.fieldnames or [])), None)
    y_name = next((name for name in y_fields if name in (reader.fieldnames or [])), None)
    if x_name is None or y_name is None:
        return {"available": False, "source": str(path), "points": [], "message": "EE X/Y fields unavailable"}
    points = []
    for row in rows:
        x = _finite_float(row.get(x_name))
        y = _finite_float(row.get(y_name))
        if x is not None and y is not None:
            points.append({"x": x, "y": y})
    if len(points) > limit:
        stride = max(1, len(points) // limit)
        points = points[::stride]
    return {
        "available": bool(points),
        "source": str(path),
        "points": points,
        "fields": {"x": x_name, "y": y_name},
        "message": "" if points else "no finite trajectory samples",
    }


class TrajectoryCatalog:
    """Low-rate cached discovery; never scan files from a control callback."""

    def __init__(self, roots: tuple[Path, ...], ttl_s: float = 1.0) -> None:
        self.roots = roots
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._updated = 0.0
        self._latest: Path | None = None

    def latest(self) -> Path | None:
        with self._lock:
            now = time.monotonic()
            if now - self._updated < self.ttl_s:
                return self._latest
            candidates: list[Path] = []
            for root in self.roots:
                if root.is_dir():
                    candidates.extend(root.glob("*.csv"))
                    candidates.extend(root.glob("**/debug_steps.csv"))
            files = [path for path in candidates if path.is_file()]
            self._latest = max(files, key=lambda item: item.stat().st_mtime, default=None)
            self._updated = now
            return self._latest
