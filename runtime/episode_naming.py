"""Outcome-grouped raw episode naming, sequencing, and atomic finalization.

New recordings are stored as:

    <raw_root>/success/<NNN_HHMM>/
    <raw_root>/failure/<NNN_HHMM>/

``NNN`` is a single chronological attempt sequence shared by both outcome
folders (so ``success/001_...``, ``failure/002_...``, ``success/003_...`` is
the expected interleaving), and ``HHMM`` is the local hour+minute the episode
was saved, purely for human readability in a directory listing. Full
provenance (date, exact time, task, outcome, dataset version, attempt index)
stays in that episode's ``metadata.json``.

Diagnostic saves and historical flat episodes (``<raw_root>/<old_name>/``)
are unrelated to this module: they do not consume a sequence number and are
handled by the caller with the pre-existing flat-naming path.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

OUTCOME_DIRS = ("success", "failure")

# Matches the spec's discovery pattern exactly: at least 3 digits, then an
# underscore, then exactly 4 digits (HHMM). Deliberately permissive about
# the numeric ranges of HH/MM here -- this is used to *discover* existing
# grouped episodes, not to validate a freshly formatted name.
EPISODE_NAME_PATTERN = re.compile(r"^(\d{3,})_(\d{4})$")


def is_grouped_episode_name(name: str) -> bool:
    """True if ``name`` matches the current NNN_HHMM outcome-grouped scheme."""
    return EPISODE_NAME_PATTERN.match(name) is not None


def format_episode_name(sequence: int, hhmm: str) -> str:
    """Build a ``NNN_HHMM`` directory name.

    ``sequence`` is zero-padded to at least 3 digits and expands naturally
    past 999 rather than truncating. ``hhmm`` must already be 4 digits.
    """
    if sequence < 1:
        raise ValueError(f"sequence must be >= 1, got {sequence}")
    if len(hhmm) != 4 or not hhmm.isdigit():
        raise ValueError(f"hhmm must be exactly 4 digits, got {hhmm!r}")
    return f"{sequence:03d}_{hhmm}"


def next_attempt_sequence(raw_root: Path) -> int:
    """Next global attempt number for a NEW success/failure episode.

    Scans both ``raw_root/success`` and ``raw_root/failure`` for directories
    whose name matches ``NNN_HHMM``. The result is one past the highest
    sequence number found across *both* outcome folders combined -- a single
    shared namespace, not one counter per outcome. Directories that do not
    match (legacy flat episodes, diagnostic saves, stray files) are ignored.
    Returns 1 when neither outcome folder has a matching episode yet.
    """
    highest = 0
    for outcome in OUTCOME_DIRS:
        outcome_dir = raw_root / outcome
        if not outcome_dir.is_dir():
            continue
        for entry in outcome_dir.iterdir():
            if not entry.is_dir():
                continue
            match = EPISODE_NAME_PATTERN.match(entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def finalize_grouped_episode(
    scratch: Path,
    raw_root: Path,
    outcome: str,
    metadata: dict[str, Any],
    *,
    hhmm: str | None = None,
    max_attempts: int = 1000,
) -> tuple[Path, int]:
    """Atomically move a finished episode into ``raw_root/<outcome>/NNN_HHMM``.

    ``scratch`` is the fully-populated, already-finished episode directory
    (metadata.json is about to be (re)written into it, everything else is
    already there). ``metadata`` is written to ``scratch/metadata.json``
    with ``attempt_index`` set to the reserved sequence number, then
    ``scratch`` is moved into place with :func:`shutil.move` -- a rename
    when possible, so no half-written episode is ever visible under
    ``raw_root``.

    The target existence check and the move are not a single atomic OS call,
    so a collision (another process claimed the same number and minute
    between the two) is handled by recomputing a fresh sequence number and
    retrying, up to ``max_attempts`` times, rather than overwriting anything
    or raising immediately.
    """
    if outcome not in OUTCOME_DIRS:
        raise ValueError(f"outcome must be one of {OUTCOME_DIRS}, got {outcome!r}")
    if not scratch.is_dir():
        raise FileNotFoundError(f"scratch episode directory does not exist: {scratch}")
    resolved_hhmm = hhmm if hhmm is not None else time.strftime("%H%M")
    outcome_dir = raw_root / outcome
    outcome_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = scratch / "metadata.json"
    for _ in range(max_attempts):
        sequence = next_attempt_sequence(raw_root)
        name = format_episode_name(sequence, resolved_hhmm)
        target = outcome_dir / name
        if target.exists():
            continue
        stamped = dict(metadata, attempt_index=sequence)
        metadata_path.write_text(
            json.dumps(stamped, indent=2) + "\n", encoding="utf-8"
        )
        try:
            shutil.move(str(scratch), str(target))
        except (FileExistsError, shutil.Error, OSError):
            # Another process claimed this exact name between the check
            # above and the move. Recompute and try again rather than
            # risking shutil.move's "nest inside an existing directory"
            # behavior or overwriting a finalized episode.
            continue
        return target, sequence
    raise RuntimeError(
        f"unable to reserve a unique {outcome} episode directory under "
        f"{raw_root} after {max_attempts} attempts"
    )
