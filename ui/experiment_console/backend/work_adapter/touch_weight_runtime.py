"""Normal Touch collection: operator-selected folder and unlimited episode counts."""
from dataclasses import replace
import time
import uuid
from .runtime import WorkRuntime
from runtime.collection_counts import counts, event, validate_output
from runtime.collection_preferences import load_output, save_output, PREFERENCES


class TouchWeightRuntime(WorkRuntime):
    def __init__(self, *, preferences=PREFERENCES):
        self._preferences = preferences
        self._selected_output = load_output(preferences)
        self._counts_cache = None
        self._counts_at = 0.
        self._run_id = None
        super().__init__()
        self._message = ("Previous Dataset Output folder restored."
                         if self._selected_output else "Choose a Dataset Output folder, then START RECORDING.")

    def _collection_progress(self):
        return None  # There is no collection quota or automatic completion.

    def set_output_folder(self, path):
        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise ValueError("Finish the current recording before changing its folder")
            self._selected_output = save_output(path, self._preferences)
            self._counts_at = 0.
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            value=super().snapshot()
            if self._run_id and self._process is not None and not self._process.is_alive():
                event(self._config.dataset_output_root,self._run_id,'finished',
                      worker_exit_code=self._process.exitcode)
                self._run_id=None
                self._counts_at=0.
            if self._selected_output is not None and time.monotonic()-self._counts_at>.5:
                self._counts_cache=counts(self._selected_output)
                self._counts_at=time.monotonic()
            value['collection_stats']=self._counts_cache
            value['selected_output_folder']=str(self._selected_output) if self._selected_output else None
            value['collection']=None
            value['manual_collection']=True
            return value

    def start(self, config):
        with self._lock:
            root=validate_output(config.dataset_output_root)
            if config.collection_plan_id:
                raise ValueError("Manual collection has no fixed plan")
            if self._process is not None and self._process.is_alive():
                raise ValueError("A recording is already running")
            self.snapshot()  # finalize accounting of the previous worker first
            self._selected_output=save_output(root, self._preferences)
            run_id=uuid.uuid4().hex
            summary=counts(root)
            result=super().start(replace(config,seed=summary['success'],episodes=1,
                dataset_output_root=str(root),collection_run_id=run_id))
            self._run_id=run_id
            event(root,run_id,'started',worker_pid=self._process.pid)
            self._counts_at=0.
            return self.snapshot()
