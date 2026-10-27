"""Small, dependency-free contracts shared by console adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    available: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobRequest:
    kind: str
    command: tuple[str, ...]
    cwd: str
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectorySummary:
    path: str
    rows: int
    columns: tuple[str, ...]
