"""Mimic environments for robot manipulation."""

from mimic.envs.base import MimicEnv
from mimic.envs.registry import list_envs, make, register

__all__ = ["MimicEnv", "make", "list_envs", "register"]
