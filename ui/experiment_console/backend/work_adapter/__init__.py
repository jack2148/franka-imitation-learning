"""WORK-mode adapter and its background orchestration runtime."""

from ..contracts import AdapterStatus
from ..paths import policy_learning_root, robot_system_root
from .runtime import WorkConfig, WorkRuntime


class WorkAdapter:
    """Expose real collection state without owning robot or dataset logic."""

    haptic_available = False
    haptic_enabled = False

    def __init__(self) -> None:
        from ..task_profiles import launch_leader
        if launch_leader() not in {"omy", "geomagic"}:
            raise ValueError("This public snapshot supports OMY and Geomagic Touch")
        from runtime.touch_weight_collection import enabled as weight_collection_enabled
        if weight_collection_enabled():
            from .touch_weight_runtime import TouchWeightRuntime
            self.runtime = TouchWeightRuntime()
        else:
            self.runtime = WorkRuntime()

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name="work",
            available=robot_system_root().is_dir() and policy_learning_root().is_dir(),
            message="Collection adapter boundary is ready; no hardware process was started.",
            details={
                "haptic_available": self.haptic_available,
                "haptic_enabled": self.haptic_enabled,
            },
        )

    def state(self) -> dict:
        return self.runtime.snapshot()

    def start(self, config: WorkConfig) -> dict:
        return self.runtime.start(config)

    def action(self, action: str) -> dict:
        return self.runtime.dispatch(action)

    def frame_bytes(self, camera: str = "task") -> bytes | None:
        return self.runtime.frame_bytes(camera)

    def close(self) -> None:
        self.runtime.close()
