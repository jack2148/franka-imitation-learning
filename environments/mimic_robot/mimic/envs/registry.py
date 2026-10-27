from __future__ import annotations

from mimic.config.models import EnvConfig
from mimic.envs.base import MimicEnv

_REGISTRY: dict[str, type[MimicEnv]] = {}


def register(name: str):
    """Decorator to register an environment class."""

    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def make(name: str, **kwargs) -> MimicEnv:
    """Create an environment by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown env '{name}'. Available: {available}")
    config = EnvConfig(name=name, **kwargs)
    return _REGISTRY[name](config)


def list_envs() -> list[str]:
    """List all registered environment names."""
    return sorted(_REGISTRY.keys())
