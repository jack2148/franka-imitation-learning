"""Resolve workspace paths without importing either project."""

from pathlib import Path
import os
import sys


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def robot_system_root() -> Path:
    return workspace_root() / "teleop" / "omy_fr3"


def policy_learning_root() -> Path:
    # Policy artifacts and experiment metadata live in this checkout by
    # default.  Keep the environment override for deployments that still use
    # a separate policy workspace.
    return Path(os.environ.get("MIMIC_POLICY_ROOT", str(workspace_root())))


def lerobot_root() -> Path:
    return Path(os.environ.get("MIMIC_LEROBOT_ROOT", str(workspace_root() / "vendor/lerobot")))


def shared_datasets_root() -> Path:
    return workspace_root() / "data" / "raw"


def shared_runs_root() -> Path:
    from runtime.collection_workspace import research_root
    if research_root() is not None:
        return research_root() / "runs"
    return workspace_root() / "runs"


def ensure_policy_import_path() -> None:
    """Expose the linked policy repository to adapters without hardcoding it."""
    root = policy_learning_root()
    lerobot = lerobot_root() / "src"
    for path in (root, lerobot):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
