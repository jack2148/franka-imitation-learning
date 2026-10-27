"""Cached discovery of completed evaluation run directories."""

from pathlib import Path
import time


class RunIndex:
    def __init__(self, root: Path, ttl: float = 3.0) -> None:
        self.root, self.ttl = root, ttl
        self._expires = 0.0
        self._runs: list[Path] = []

    def paths(self) -> list[Path]:
        now = time.monotonic()
        if now < self._expires:
            return self._runs
        paths = []
        if self.root.is_dir():
            for path in self.root.iterdir():
                if path.is_dir() and (path / "evaluation_config.json").is_file():
                    paths.append(path)
        self._runs = sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
        self._expires = now + self.ttl
        return self._runs
