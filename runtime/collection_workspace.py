"""Explicit Touch research workspace, isolated from frozen collection plans."""
from __future__ import annotations

import os
from pathlib import Path

COLLECTION_WORKSPACES = Path(__file__).resolve().parents[1] / "outputs" / "research"


def research_root() -> Path | None:
    value = os.environ.get("MIMIC_COLLECTION_WORKSPACE")
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    base = COLLECTION_WORKSPACES.resolve()
    if root == base or base not in root.parents:
        raise ValueError(f"Touch workspace must be a session directory under {base}")
    return root


def require_research_output(path: str | Path) -> Path:
    root = research_root()
    if root is None:
        raise ValueError("Touch collection requires the explicit ui_geo_sq launcher")
    selected = Path(path).expanduser().resolve()
    raw = (root / "raw").resolve()
    if raw == root or root not in raw.parents:
        raise ValueError("Touch raw directory must not link outside its research session")
    if selected != raw and raw not in selected.parents:
        raise ValueError("Touch collection must remain inside this session's raw directory")
    return selected
