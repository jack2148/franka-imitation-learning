"""Safe folder browsing and dataset inspection under experiments/."""
from pathlib import Path
import os
import re
import shutil

from runtime.dataset_validation import validate_dataset
from .paths import workspace_root
from runtime.collection_workspace import research_root

EXPERIMENTS_ROOT = (Path("/") if os.environ.get("MIMIC_WEIGHT_COLLECTION") == "1"
                    else research_root() or (workspace_root() / "experiments").resolve())
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def inside(path: str | None = None, *, must_exist: bool = False) -> Path:
    raw = Path(path).expanduser() if path else EXPERIMENTS_ROOT
    candidate = raw if raw.is_absolute() else EXPERIMENTS_ROOT / raw
    resolved = candidate.resolve(strict=False)
    if resolved != EXPERIMENTS_ROOT and EXPERIMENTS_ROOT not in resolved.parents:
        raise ValueError("path must remain under the experiments workspace")
    if must_exist and not resolved.is_dir():
        raise ValueError("selected path is not an existing directory")
    return resolved


def folder_listing(path: str | None = None) -> dict:
    current = inside(path, must_exist=True)
    children = []
    for item in sorted(current.iterdir(), key=lambda p: p.name.lower()):
        if not item.is_dir():
            continue
        try:
            resolved = inside(str(item), must_exist=True)
        except ValueError:
            continue
        children.append({"name": item.name, "path": str(resolved), "relative": str(resolved.relative_to(EXPERIMENTS_ROOT))})
    usage = shutil.disk_usage(current)
    return {"root": str(EXPERIMENTS_ROOT), "current": str(current),
            "relative": str(current.relative_to(EXPERIMENTS_ROOT)),
            "parent": str(current.parent) if current != EXPERIMENTS_ROOT else None,
            "readable": os.access(current, os.R_OK), "writable": os.access(current, os.W_OK), "free_bytes": usage.free,
            "children": children}


def create_folder(parent: str | None, name: str) -> dict:
    if not _NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError("invalid folder name")
    target_parent = inside(parent, must_exist=True)
    target = target_parent / name
    if target.exists():
        raise FileExistsError("folder already exists")
    target.mkdir()
    return folder_listing(str(target_parent))


def inspect_path(path: str | None) -> dict:
    selected = inside(path, must_exist=True)
    listing = folder_listing(str(selected))
    listing["dataset"] = validate_dataset(selected)
    return listing
