"""WORK runtime isolated from the HTTP server process."""
from __future__ import annotations
import sys
from collections import deque
from dataclasses import asdict, dataclass, replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time
from typing import Any
from ..paths import ensure_policy_import_path, policy_learning_root, robot_system_root, shared_datasets_root, workspace_root
from ..paths import shared_runs_root
from ..trajectory_reader import TrajectoryCatalog, read_xy_trajectory
from ..common.camera_projection import task_region_overlay
from ..experiment_paths import folder_listing, inside
from ..task_profiles import canonical_task, launch_geomagic_source, launch_profile, launch_leader, purpose_defaults
from runtime.collection_plan import collection_plan
from runtime.ui_camera_layout import geomagic_square_front_panel, front_camera_state
from runtime.collection_workspace import research_root, require_research_output
REGIONS = ('A', 'B', 'C')

@dataclass(frozen=True)
class WorkConfig:
    region: str = 'A'
    task: str = 'fr3-pick-place'
    input_backend: str = 'zmq'
    omy_endpoint: str = 'tcp://127.0.0.1:5557'
    episodes: int = 1
    seed: int = 0
    operator_id: str = 'operator_01'
    dataset_version: str = 'omy_fr3_mimic_20hz_diagnostic_v1'
    dataset_output_root: str | None = None
    collection_plan_id: str | None = None
    workspace_validation: dict | None = None
    preflight_candidate: dict | None = None
    geomagic_source: str | None = None
    geomagic_device: dict[str, Any] | None = None
    collection_run_id: str | None = None

def extract_ee_position(pose: Any) -> tuple[float, float, float]:
    """Extract xyz from TeleopRunner's [qw,qx,qy,qz,x,y,z] pose format."""
    values = list(pose)
    if len(values) != 7:
        raise ValueError(f'expected 7-value EE pose, got {len(values)}')
    return (float(values[4]), float(values[5]), float(values[6]))

def _next_square_demo_seed(raw_root: Path, batch_size: int=50) -> int:
    """Return the next seed after the successful square demos.

    A failed/aborted attempt must be retried with the same initial state.  The
    seed is therefore consumed only by a metadata record whose outcome is
    ``success``.  ``collection_seed`` is accepted for older recordings, while
    ``distribution_seed`` is the current schema used by the active worker.
    """
    used: set[int] = set()
    if raw_root.is_dir():
        for metadata_path in raw_root.rglob('metadata.json'):
            try:
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                if metadata.get('initial_state_distribution') != 'square_uniform_eval_v1':
                    continue
                if str(metadata.get('outcome', '')).lower() != 'success':
                    continue
                value = metadata.get('distribution_seed')
                if not isinstance(value, int):
                    value = metadata.get('collection_seed')
                if isinstance(value, int) and value >= 0:
                    used.add(value)
            except (OSError, ValueError):
                continue
    for seed in range(batch_size):
        if seed not in used:
            return seed
    return max(used) + 1 if used else 0

class WorkRuntime:
    """Non-blocking parent facade around one MuJoCo/control worker process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._context = mp.get_context('spawn')
        self._commands = self._context.Queue()
        self._states = self._context.Queue(maxsize=1)
        self._process = None
        self._omy_publisher: subprocess.Popen[str] | None = None
        self._mujoco_process: subprocess.Popen[str] | None = None
        self._mujoco_log = Path(os.environ.get('MIMIC_MUJOCO_LOG', '/tmp/mimic-mujoco-viewer.log'))
        self._teleop_probe_at = 0.0
        self._teleop_available = False
        self._config: WorkConfig | None = None
        self._last_state: dict[str, Any] | None = None
        self._phase = 'IDLE'
        self._message = 'No live collection runtime is running.'
        self._inventory = InventoryCache()
        self._trajectory_path: Path | None = None
        self._trajectory_value: dict[str, Any] | None = None
        self._catalog = TrajectoryCatalog((robot_system_root() / 'logs', policy_learning_root() / 'outputs' / 'teleop', policy_learning_root() / 'outputs' / 'eval'), ttl_s=10.0)
        self._frame_dir = shared_runs_root() / 'work' / 'live'
        self._launch_profile = launch_profile()
        self.leader = launch_leader()
        self.geomagic_source = launch_geomagic_source()
        self._front_panel = geomagic_square_front_panel(self.leader, self.geomagic_source, self._launch_profile.task)
        self._default_raw_root = purpose_defaults(self._launch_profile)['dataset_output']
        if self.leader == 'geomagic':
            inside(self._default_raw_root).mkdir(parents=True, exist_ok=True)
        self._geomagic_device = None
        self._haptics_enabled = self.leader == 'geomagic' and self.geomagic_source == 'physical_hdapi' and (self._launch_profile.task in {'mimicgen_square'}) and (os.environ.get('MIMIC_GEOMAGIC_HAPTICS', '1') == '1')
        self._sync_root = Path(os.environ['GEOMAGIC_SYNC_CAPTURE']).expanduser().resolve() if self.geomagic_source == 'physical_hdapi' and os.environ.get('GEOMAGIC_SYNC_CAPTURE') else None
        self._sync_capture: dict[str, Any] | None = None
        self._sync_thread: threading.Thread | None = None
        if self.geomagic_source == 'physical_hdapi':
            from .geomagic_device import GeomagicDeviceSupervisor, GeomagicHapticSupervisor
            self._geomagic_device = GeomagicHapticSupervisor() if self._haptics_enabled else GeomagicDeviceSupervisor()
            self._geomagic_device.start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_teleop_status()
            self._drain_states()
            if self._process is not None and (not self._process.is_alive()):
                self._finish_sync_capture()
            if self._process is not None and (not self._process.is_alive()) and (self._last_state is not None) and (self._last_state.get('phase') not in {'IDLE', 'ERROR'}):
                (self._phase, self._message) = ('ERROR', 'WORK worker exited before reporting a terminal state.')
            if self._last_state is not None:
                result = self._idle_snapshot(self._historical_trajectory())
                result.update(self._last_state)
                if self._geomagic_device is not None and isinstance(result.get('status'), dict):
                    status = dict(result['status'])
                    leader = dict(status.get('leader') or {})
                    leader['device'] = self._geomagic_device.snapshot()
                    status['leader'] = leader
                    result['status'] = status
                if self._haptics_enabled:
                    result['haptic'] = self._haptic_snapshot(result.get('haptic'))
                result.setdefault('status', self._idle_snapshot({'available': False, 'source': None, 'points': [], 'message': 'No trajectory available'})['status'])
                result['counts'] = self._inventory.value()
                result['grasp_guidance'] = self._grasp_guidance_snapshot()
                result['collection'] = self._collection_progress()
                result['camera'] = dict(result.get('camera') or {})
                result['camera']['collection_boundary'] = self._collection_boundary_snapshot()
                result['latency_capture'] = dict(self._sync_capture) if self._sync_capture else None
                result['workspace_validation'] = self._workspace_validation_snapshot()
                return result
            result = self._idle_snapshot(self._historical_trajectory())
            result['latency_capture'] = dict(self._sync_capture) if self._sync_capture else None
            result['workspace_validation'] = self._workspace_validation_snapshot()
            return result

    def _workspace_validation_snapshot(self):
        name = os.environ.get('MIMIC_UI_WORKSPACE_VALIDATION')
        if not (name and self.leader == 'geomagic' and (self.geomagic_source == 'physical_hdapi') and (self._launch_profile.task == 'mimicgen_square')):
            return None
        from runtime.workspace_validation import load_validation, summary
        result = summary(load_validation(name))
        result['active_point'] = self._config.workspace_validation['point'] if self._config and self._config.workspace_validation and self._process and self._process.is_alive() else None
        return result

    def _collection_progress(self) -> dict[str, Any] | None:
        if os.environ.get('MIMIC_UI_WORKSPACE_VALIDATION') and self.leader == 'geomagic' and (self.geomagic_source == 'physical_hdapi'):
            return None
        plan_id = self._config.collection_plan_id if self._config else self._launch_profile.collection_plan_id
        plan = collection_plan(plan_id)
        if plan is None:
            return None
        root = inside(self._config.dataset_output_root) if self._config and self._config.dataset_output_root else inside(self._default_raw_root)
        outcome = None
        if self._last_state and self._last_state.get('phase') in {'IDLE', 'ERROR'}:
            outcome = (self._last_state.get('task_status') or {}).get('reason')
            if self._last_state.get('phase') == 'ERROR' and outcome == 'success' and (self._config is not None) and (self._config.seed not in plan.accepted_seeds(root)):
                outcome = 'success_not_saved'
            elif self._last_state.get('phase') == 'ERROR':
                outcome = None
        progress = plan.progress(root, last_outcome=outcome)
        if self._last_state and self._last_state.get('phase') not in {'IDLE', 'ERROR'}:
            attempt = self._last_state.get('collection_attempt') or {}
            if attempt.get('collection_order') == progress['current_seed'] and attempt.get('collection_plan') == plan.distribution_id:
                progress['attempt_number'] = attempt['attempt_number']
        return progress

    def start(self, config: WorkConfig) -> dict[str, Any]:
        validation = self._workspace_validation_snapshot()
        if validation:
            if validation['workspace_status'] != 'DRAFT':
                raise ValueError('workspace already frozen; restart normal collection UI')
            config = replace(config, collection_plan_id=None, workspace_validation=dict(plan_name=validation['plan_name'], point=validation['point']['name']))
        elif config.workspace_validation:
            raise ValueError('validation payload requires the explicit physical validation launcher')
        if config.region not in REGIONS:
            raise ValueError('region must be A, B, or C')
        if canonical_task(config.task) != self._launch_profile.task:
            raise ValueError(f'recording blocked: UI task identity {self._launch_profile.task} does not match requested runtime task {canonical_task(config.task)}')
        if config.input_backend not in {'ros', 'zmq', 'synthetic', 'geomagic'}:
            raise ValueError('input_backend must be ros, zmq, synthetic, or geomagic')
        if (config.input_backend == 'geomagic') != (self.leader == 'geomagic'):
            raise ValueError('input backend does not match the launcher-selected leader')
        if self.leader == 'geomagic':
            root = inside(self._default_raw_root)
            selected = inside(config.dataset_output_root) if config.dataset_output_root else root
            if selected != root and root not in selected.parents and (os.environ.get('MIMIC_WEIGHT_COLLECTION') != '1'):
                raise ValueError('Geomagic recordings must stay under the Geomagic raw root')
            config = replace(config, dataset_output_root=str(selected), dataset_version=self._launch_profile.dataset_version, geomagic_source=self.geomagic_source, geomagic_device=None)
        else:
            config = replace(config, geomagic_source=None, geomagic_device=None)
        xy_requested = self.leader == 'geomagic' and (config.collection_plan_id or self._launch_profile.collection_plan_id or '').startswith('square_geomagic_xy50_v')
        if config.dataset_output_root and (not xy_requested):
            selected = inside(config.dataset_output_root, must_exist=True)
            if not folder_listing(str(selected))['writable']:
                raise ValueError('dataset output directory is not writable')
        with self._lock:
            self._drain_states()
            if self._sync_capture and self._sync_capture['state'] in {'recording', 'saving'}:
                self._finish_sync_capture() if self._process is not None and (not self._process.is_alive()) else None
                raise RuntimeError('previous synchronization capture is still saving')
            if self._process is not None and self._process.is_alive():
                raise RuntimeError('collection runtime is already active')
            plan_id = config.collection_plan_id or self._launch_profile.collection_plan_id
            plan = collection_plan(plan_id)
            if plan is not None:
                from runtime.collection_plan import XYCollectionPlan
                if isinstance(plan, XYCollectionPlan):
                    if self.leader != 'geomagic' or self.geomagic_source != 'physical_hdapi' or self._launch_profile.collection_plan_id != plan_id:
                        raise ValueError('XY50 requires explicit selection in physical ui_geo_sq')
                    plan.require_frozen()
                    selected = inside(config.dataset_output_root)
                    plan.ensure_isolated_root(selected)
                    selected.mkdir(parents=True, exist_ok=True)
                    if not folder_listing(str(selected))['writable']:
                        raise ValueError('dataset output directory is not writable')
                if plan.task != canonical_task(config.task):
                    raise ValueError('collection plan does not match requested task')
                output_root = inside(config.dataset_output_root, must_exist=True) if config.dataset_output_root else inside(plan.default_raw_root)
                plan.ensure_isolated_root(output_root)
                next_seed = plan.next_seed(output_root)
                if next_seed is None:
                    raise RuntimeError(f'{plan.label}: COLLECTION COMPLETE ({plan.target}/{plan.target})')
                config = replace(config, collection_plan_id=plan_id, dataset_output_root=str(output_root), seed=next_seed, episodes=plan.target)
            if self._geomagic_device is not None:
                config = replace(config, geomagic_device=self._physical_preflight())
            self._commands = self._context.Queue()
            self._states = self._context.Queue(maxsize=1)
            (self._config, self._last_state) = (config, None)
            (self._phase, self._message) = ('STARTING', 'Starting isolated WORK worker process.')
            if self._geomagic_device is not None:
                self._message = 'Starting Geomagic WORK; verifying fresh physical Touch packets on 127.0.0.1:5558.'
            elif self.leader == 'geomagic':
                self._message = 'Starting Geomagic WORK; waiting for synthetic UDP on 127.0.0.1:5558.'
            elif config.input_backend == 'zmq':
                self._start_omy_publisher(config)
            session = f'session_{os.getpid()}_{time.time_ns()}'
            self._frame_dir = shared_runs_root() / 'work' / 'live' / session
            worker_config = asdict(config)
            if self._haptics_enabled:
                worker_config.update(self._geomagic_device.worker_transport())
            worker_config['_preview_dir'] = str(self._frame_dir)
            worker_config['_preview_image_size'] = [int(os.environ.get('MIMIC_UI_PREVIEW_HEIGHT', '960')), int(os.environ.get('MIMIC_UI_PREVIEW_WIDTH', '1280'))]
            if self._sync_root is not None:
                directory = self._sync_root / f"work_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
                directory.mkdir(parents=True, exist_ok=False)
                from geomagic.teleop.sync_capture import SyncCapture
                SyncCapture(directory, seconds=float(os.environ.get('GEOMAGIC_SYNC_SECONDS', '60')), capacity=int(os.environ.get('GEOMAGIC_SYNC_RECORDS', '200000')))
                pid = self._geomagic_device.begin_capture(directory)
                worker_config['_sync_capture_directory'] = str(directory)
                self._sync_capture = dict(directory=str(directory), state='recording', error=None, native_pid=pid)
            self._process = self._context.Process(target=work_worker_main, args=(worker_config, self._commands, self._states), daemon=False)
            try:
                self._process.start()
            except BaseException:
                self._process = None
                self._finish_sync_capture()
                raise
            return self.snapshot()

    def set_weight(self, value, recording_id, request_id) -> dict[str, Any]:
        from runtime.sample_weight import check_weight
        check_weight(value)
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 128:
            raise ValueError('weight request_id is required (maximum 128 characters)')
        with self._lock:
            self._drain_states()
            state = self._last_state or {}
            if self._process is None or not self._process.is_alive() or (not state.get('recording')):
                raise ValueError('Start recording before selecting a weight')
            if recording_id != (state.get('weight') or {}).get('recording_id') or not recording_id:
                raise ValueError('Recording changed; refresh weight state')
            self._commands.put(dict(command='weight', weight=value, recording_id=recording_id, request_id=request_id))
            return dict(queued=True, request_id=request_id, recording_id=recording_id)

    def dispatch(self, action: str) -> dict[str, Any]:
        if self._config and self._config.workspace_validation and (action in {'success', 'failure', 'stop_and_save'}):
            raise ValueError('workspace validation never saves episodes; STOP then record assessment with helper')
        if action not in {'stop_and_save', 'success', 'failure', 'discard', 'abort', 'reset', 'reset_initial'}:
            raise ValueError(f'unsupported work action: {action}')
        with self._lock:
            if self._process is None or not self._process.is_alive():
                raise RuntimeError('no live collection runtime is active')
            self._commands.put(action)
            if self._haptics_enabled:
                self._geomagic_device.zero_force()
            if action != 'reset_initial':
                self._finish_sync_capture()
            return self.snapshot()

    def _finish_sync_capture(self) -> None:
        """Background disk work after control stops; HTTP polling stays responsive."""
        if not self._sync_capture or self._sync_capture['state'] != 'recording':
            return
        capture = self._sync_capture
        capture['state'] = 'saving'
        process = self._process

        def save() -> None:
            directory = Path(capture['directory'])
            try:
                deadline = time.monotonic() + 120.0
                while not list(directory.glob('host-*.jsonl.gz')):
                    if process is None or not process.is_alive():
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('WORK did not finish its host capture within 120 seconds')
                    time.sleep(0.05)
                self._geomagic_device.save_capture(directory, capture['native_pid'])
                hosts = list(directory.glob('host-*.jsonl.gz'))
                if not hosts:
                    raise RuntimeError('host capture is missing; native capture was checkpointed')
                import gzip
                with gzip.open(hosts[0], 'rt') as stream:
                    metadata = json.loads(stream.readline())
                roles = ['preview']
                if metadata['settings'].get('recording_enabled'):
                    roles.append('recorder')
                deadline = time.monotonic() + 120.0
                while any((not list(directory.glob(role + '-*.jsonl.gz')) for role in roles)):
                    if process is None or not process.is_alive() or time.monotonic() >= deadline:
                        raise RuntimeError('camera capture logs are incomplete; host/native logs are saved')
                    time.sleep(0.05)
                with self._lock:
                    capture['state'] = 'saved'
                    capture['stopped'] = metadata['stopped']
            except Exception as error:
                with self._lock:
                    (capture['state'], capture['error']) = ('error', str(error))
        self._sync_thread = threading.Thread(target=save, name='geomagic-capture-save', daemon=True)
        self._sync_thread.start()

    def set_teleop_mode(self, mode: Any) -> dict[str, Any]:
        """Request a runtime-owned Cartesian control mode."""
        value = str(mode).lower()
        if self.leader == 'geomagic' and value == 'rotation_3d':
            value = 'rotation'
        allowed = {'position', 'rotation', 'rotation_z', '6dof'} if self.leader == 'geomagic' else {'position', 'rotation'}
        if value not in allowed:
            raise ValueError('teleop mode must be position, rotation_z, rotation (3D) or 6dof' if self.leader == 'geomagic' else 'teleop mode must be position or rotation')
        with self._lock:
            if self._process is None or not self._process.is_alive():
                raise RuntimeError('start WORK before selecting a teleop mode')
            self._commands.put(f'teleop_mode:{value}')
            return {'ok': True, 'requested_mode': value}

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.is_alive():
                self._commands.put('abort')
            self._finish_sync_capture()
        if self._haptics_enabled:
            self._geomagic_device.stop()
        if process is not None:
            process.join(timeout=120 if self._sync_capture else 8)
            if process.is_alive():
                import psutil
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.Error:
                    children = []
                process.terminate()
                process.join(timeout=2)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.Error:
                        pass
                (_, alive) = psutil.wait_procs(children, timeout=2)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=32)
        with self._lock:
            self._process = None
            self._stop_mujoco()
            self._stop_omy_publisher()
        if self._geomagic_device is not None and (not self._haptics_enabled):
            self._geomagic_device.stop()

    def _physical_preflight(self) -> dict[str, Any]:
        """Refuse a physical START WORK unless our own streamer is healthy."""
        from .geomagic_device import STREAM_PORT, udp_port_free
        (ready, reason) = self._geomagic_device.ready()
        if not ready:
            raise RuntimeError(f'START WORK blocked: {reason}')
        if not udp_port_free(STREAM_PORT):
            raise RuntimeError(f'START WORK blocked: UDP 127.0.0.1:{STREAM_PORT} is already bound by another process (for example csv_capture.py or validate_lan_stream); stop it first')
        device = self._geomagic_device.snapshot()
        return {key: device[key] for key in ('device_source', 'model', 'serial', 'session_id', 'streamer_pid', 'hdapi_hz', 'publish_hz', 'calibration', 'force_output_enabled')}

    def mujoco_state(self) -> dict[str, Any]:
        with self._lock:
            if self.leader == 'geomagic':
                return {'available': False, 'running': False, 'pid': None, 'teleop_connected': False, 'endpoint': 'udp://127.0.0.1:5558', 'log': str(self._mujoco_log), 'reason': 'Geomagic uses the WORK simulation.'}
            self._refresh_teleop_status()
            running = self._mujoco_process is not None and self._mujoco_process.poll() is None
            return {'running': running, 'pid': self._mujoco_process.pid if running else None, 'teleop_connected': self._teleop_available, 'endpoint': 'tcp://127.0.0.1:5557', 'log': str(self._mujoco_log)}

    def start_mujoco(self) -> dict[str, Any]:
        with self._lock:
            if self.leader == 'geomagic':
                raise RuntimeError('Standalone MuJoCo viewer is unavailable for Geomagic; start WORK')
            self._refresh_teleop_status(force=True)
            if not self._teleop_available:
                raise RuntimeError('OMY teleop is not connected. Start teleop first, then press MuJoCo START.')
            if self._mujoco_process is not None and self._mujoco_process.poll() is None:
                return self.mujoco_state()
            self._start_omy_publisher(WorkConfig(input_backend='zmq', omy_endpoint='tcp://127.0.0.1:5557'))
            bridge = workspace_root() / 'teleop' / 'omy_fr3' / 'launch' / 'FR3_omy_bridge.py'
            if not bridge.is_file():
                raise FileNotFoundError(f'MuJoCo bridge not found: {bridge}')
            env = os.environ.copy()
            env.update({'OMY_INPUT': 'zmq', 'OMY_ZMQ_ENDPOINT': 'tcp://127.0.0.1:5557', 'MUJOCO_GL': env.get('MUJOCO_GL', 'egl'), 'PYTHONPATH': f"{workspace_root()}:{env.get('PYTHONPATH', '')}"})
            self._mujoco_log.parent.mkdir(parents=True, exist_ok=True)
            log_file = self._mujoco_log.open('a', encoding='utf-8')
            python = os.environ.get('MIMIC_PYTHON', sys.executable)
            self._mujoco_process = subprocess.Popen([python, str(bridge)], cwd=str(workspace_root()), env=env, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            time.sleep(0.25)
            if self._mujoco_process.poll() is not None:
                code = self._mujoco_process.returncode
                self._mujoco_process = None
                raise RuntimeError(f'MuJoCo viewer exited immediately (code {code}); see {self._mujoco_log}')
            return self.mujoco_state()

    def stop_mujoco(self) -> dict[str, Any]:
        with self._lock:
            self._stop_mujoco()
            return self.mujoco_state()

    def set_gripper_command(self, command: Any) -> dict[str, Any]:
        """Route Geomagic to WORK; retain the existing OMY ROS ABI child.

        The UI backend may run under a different Python ABI than ROS 2 Humble,
        so keep this boundary on the existing ``ros2 topic pub`` executable.
        This does not change the recorder's 8D action schema; the running
        publisher folds the command into its existing IPC message.
        """
        if self.leader == 'geomagic':
            try:
                value = float(command)
            except (TypeError, ValueError) as error:
                raise ValueError('gripper command must be finite and in [0, 255]') from error
            if isinstance(command, bool) or not math.isfinite(value) or (not 0 <= value <= 255):
                raise ValueError('gripper command must be finite and in [0, 255]')
            with self._lock:
                if self._process is None or not self._process.is_alive():
                    raise RuntimeError('start WORK before using the Spacebar gripper control')
                self._commands.put(f'gripper:{value:g}')
                return {'ok': True, 'command': value, 'leader': 'geomagic'}
        try:
            value = int(command)
        except (TypeError, ValueError) as error:
            raise ValueError('gripper command must be 0 or 255') from error
        if value not in {0, 255}:
            raise ValueError('gripper command must be 0 or 255')
        with self._lock:
            if self._process is None or not self._process.is_alive():
                raise RuntimeError('start WORK before using the Spacebar gripper control')
            result = subprocess.run(['ros2', 'topic', 'pub', '--once', '/fr3/gripper_command', 'std_msgs/msg/UInt8', f'{{data: {value}}}'], cwd=str(workspace_root()), capture_output=True, text=True, timeout=2.0, check=False)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(f'gripper command failed: {detail or result.returncode}')
            return {'ok': True, 'topic': '/fr3/gripper_command', 'command': value}

    def _stop_mujoco(self) -> None:
        process = self._mujoco_process
        self._mujoco_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _refresh_teleop_status(self, force: bool=False) -> None:
        if self.leader == 'geomagic':
            return
        now = time.monotonic()
        if not force and now - self._teleop_probe_at < 1.0:
            return
        self._teleop_probe_at = now
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], cwd=str(workspace_root()), capture_output=True, text=True, timeout=0.8, check=False)
            self._teleop_available = result.returncode == 0 and '/leader/joint_states' in result.stdout.split()
        except (OSError, subprocess.TimeoutExpired):
            self._teleop_available = False
        if self._teleop_available:
            self._start_omy_publisher(WorkConfig(input_backend='zmq'))

    def _start_omy_publisher(self, config: WorkConfig) -> None:
        """Bridge ROS JointState into the ZMQ endpoint used by the worker.

        The console backend commonly runs under conda/Python 3.13 while ROS 2
        Humble is installed for system Python 3.10. Keep that ABI boundary in
        a child process and never import ROS into the backend process.
        """
        if self._omy_publisher is not None and self._omy_publisher.poll() is None:
            return
        script = workspace_root() / 'scripts' / 'runtime' / 'publish_omy_joint_state.py'
        if not script.is_file():
            self._message = f'OMY publisher script not found: {script}'
            return
        python = os.environ.get('OMY_ROS_PYTHON', '/usr/bin/python3')
        if not Path(python).is_file():
            self._message = f'OMY_ROS_PYTHON is not available: {python}'
            return
        env = os.environ.copy()
        env.setdefault('PYTHONUNBUFFERED', '1')
        ros_pydeps = workspace_root() / '.ros_pydeps'
        if ros_pydeps.is_dir():
            env['PYTHONPATH'] = f"{ros_pydeps}:{env.get('PYTHONPATH', '')}"
        self._omy_publisher = subprocess.Popen([python, str(script), '--endpoint', config.omy_endpoint], cwd=str(workspace_root()), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
        time.sleep(0.15)
        if self._omy_publisher.poll() is not None:
            self._omy_publisher = None
            self._message = 'Using existing OMY publisher endpoint.'
        else:
            self._message = 'OMY ROS→ZMQ publisher started; waiting for JointState.'

    def _stop_omy_publisher(self) -> None:
        process = self._omy_publisher
        self._omy_publisher = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def frame_bytes(self, camera: str='operator') -> bytes | None:
        filename = {'operator': 'operator_camera.jpg', 'front_operator': 'front_operator_camera.jpg', 'left_side': 'left_side_camera.jpg', 'oblique': 'operator_camera.jpg', 'task': 'front_operator_camera.jpg'}.get(camera)
        if self._front_panel:
            filename = {'front': 'front_camera.jpg'}.get(camera, filename)
        if filename is None:
            raise ValueError('camera must be a configured preview camera (operator, front_operator, left_side, oblique, task; ui_geo_sq also front)')
        try:
            return (self._frame_dir / filename).read_bytes()
        except OSError:
            return None

    def _drain_states(self) -> None:
        while True:
            try:
                self._last_state = self._states.get_nowait()
            except queue.Empty:
                break
        if self._last_state:
            self._phase = str(self._last_state.get('phase', self._phase))
            self._message = str(self._last_state.get('message', self._message))
            if self._last_state.get('inventory_changed'):
                self._inventory.invalidate()

    def _grasp_guidance_snapshot(self) -> dict[str, Any]:
        unavailable = {'available': False, 'state': 'UNAVAILABLE', 'label': 'UNAVAILABLE', 'corrections': []}
        if self._process is None or not self._process.is_alive():
            return unavailable
        try:
            guidance = json.loads((self._frame_dir / 'grasp_guidance.json').read_text())
            age = time.time() - float(guidance['wall_time'])
            if not 0 <= age <= 0.25:
                return unavailable
            return guidance
        except (OSError, ValueError, KeyError, TypeError):
            return unavailable

    def _collection_boundary_snapshot(self) -> dict[str, Any] | None:
        if self._process is None or not self._process.is_alive():
            return None
        try:
            boundary = json.loads((self._frame_dir / 'collection_boundary.json').read_text())
            plan = collection_plan(self._config.collection_plan_id if self._config else None)
            if plan is None or boundary.get('camera') != 'operator' or len(boundary.get('polygon', [])) != 4 or (tuple((tuple(axis) for axis in boundary.get('world_xy_bounds', []))) != plan.world_xy_bounds):
                return None
            return boundary
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _idle_snapshot(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        teleop_status = _status('connected', 'ROS /leader/joint_states is available') if self._teleop_available else _status('unavailable', 'start OMY teleop first')
        device = None
        if self._geomagic_device is not None:
            device = self._geomagic_device.snapshot()
            teleop_status = _status(device['state'], f"Geomagic Touch {device['detail']}")
        elif self.leader == 'geomagic':
            teleop_status = _status('unavailable', 'Start WORK; waiting for Geomagic synthetic UDP on 127.0.0.1:5558')
        leader_status = dict(teleop_status, name=self.leader, label='Geomagic Touch' if self.leader == 'geomagic' else 'OMY-L100', connected=device['state'] == 'connected' if device else self._teleop_available, fresh=False, active=False, mode='position')
        if self.leader == 'geomagic':
            leader_status['device_source'] = self.geomagic_source
        if device is not None:
            leader_status['device'] = device
        mujoco_running = self._mujoco_process is not None and self._mujoco_process.poll() is None
        result = {'phase': self._phase, 'message': self._message, 'teleop_mode': 'position', 'mode_change_events': [], 'active_region': self._config.region if self._config else None, 'task': canonical_task(self._config.task) if self._config else self._launch_profile.task, 'dataset_root': None, 'episode': None, 'frame': None, 'elapsed_s': None, 'dataset_output_root': self._config.dataset_output_root if self._config else None, 'recording': False, 'worker_pid': None, 'status': {'leader': leader_status, 'omy': _status('unavailable', 'OMY is not selected') if self.leader == 'geomagic' else teleop_status, 'fr3': _status('connected' if mujoco_running else 'available', 'MuJoCo viewer running' if mujoco_running else 'MuJoCo viewer is stopped'), 'teleop': teleop_status, 'recording': _status('idle', self._phase), 'latency_ms': None}, 'mujoco': self.mujoco_state() if self.leader == 'geomagic' else {'running': mujoco_running, 'pid': self._mujoco_process.pid if mujoco_running else None, 'teleop_connected': self._teleop_available, 'endpoint': 'tcp://127.0.0.1:5557', 'log': str(self._mujoco_log)}, 'task_status': {'reason': None, 'success': None, 'terminated': False, 'block_goal_error_xy': None}, 'camera': {'available': False, 'message': 'No live camera frame yet.', **(front_camera_state() if self._front_panel else {})}, 'force': {'available': False, 'source': 'MuJoCo push-tool/block contact', 'fx': None, 'fy': None, 'fz': None}, 'grasp_guidance': {'available': False, 'state': 'UNAVAILABLE', 'label': 'UNAVAILABLE', 'corrections': []}, 'haptic': self._haptic_snapshot() if self.leader == 'geomagic' else {'available': False, 'enabled': False, 'message': 'Haptic is isolated in omy_franka_teleop.'}, 'trajectory': trajectory, 'control_metrics': {'available': False, 'message': 'No live control loop.'}, 'counts': self._inventory.value(), 'collection': self._collection_progress()}
        return result

    def _haptic_snapshot(self, worker=None):
        from geomagic.standalone.ui_haptics import haptic_status
        result = haptic_status(self._haptics_enabled)
        if worker:
            result.update(worker)
        if self._haptics_enabled:
            device = self._geomagic_device.snapshot()
            result.update(watchdog=device['watchdog'], rendered_force_n=device['rendered_force_n'], device_connected=device['state'] == 'connected')
        return result

    def _historical_trajectory(self) -> dict[str, Any]:
        path = self._catalog.latest()
        if path is None:
            return {'available': False, 'source': None, 'points': [], 'message': 'No trajectory available'}
        if path == self._trajectory_path and self._trajectory_value is not None:
            return self._trajectory_value
        self._trajectory_path = path
        self._trajectory_value = read_xy_trajectory(path, x_fields=('fr3_actual_position_x', 'actual_x', 'block_x'), y_fields=('fr3_actual_position_y', 'actual_y', 'block_y'))
        return self._trajectory_value

def work_worker_main(config_data: dict[str, Any], commands: Any, states: Any) -> None:
    """Child entry point. Heavy robot/control imports are intentionally here."""
    from runtime.ui_worker import work_worker_main as mimic_work_worker_main
    mimic_work_worker_main(config_data, commands, states)
    return
    config = WorkConfig(**config_data)
    frames: list[dict[str, Any]] = []
    points: deque[dict[str, float]] = deque(maxlen=600)
    periods: deque[float] = deque(maxlen=5000)
    (started, last_tick, last_publish) = (time.perf_counter(), time.perf_counter(), 0.0)
    (phase, message, dataset_root, runner) = ('STARTING', 'Loading RegionRunner in worker process.', None, None)
    initial_state = None
    initial_ee_state = None

    def publish(inventory_changed: bool=False) -> None:
        nonlocal last_publish
        now = time.perf_counter()
        if not inventory_changed and now - last_publish < 0.1:
            return
        last_publish = now
        metrics: dict[str, Any] = {'available': bool(periods), 'samples': len(periods)}
        if periods:
            ordered = sorted(periods)
            metrics.update(mean_period_s=statistics.fmean(periods), p95_period_s=ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], max_period_s=max(periods), jitter_s=statistics.pstdev(periods) if len(periods) > 1 else 0.0)
        try:
            backend = runner.backend
            state = backend.get_state()
            overlay = task_region_overlay(runner.get_object_pose().tolist(), runner.get_goal_pose().tolist(), config.region)
            states.put_nowait({'phase': phase, 'message': message, 'active_region': config.region, 'task': config.task, 'dataset_root': str(dataset_root) if dataset_root else None, 'episode': 0, 'frame': len(frames), 'elapsed_s': time.perf_counter() - started, 'recording': phase == 'RECORDING', 'worker_pid': os.getpid(), 'status': {'omy': _status('connected' if backend.input_fresh else 'unavailable', 'fresh OMY input' if backend.input_fresh else 'OMY input is stale'), 'fr3': _status('connected', 'MuJoCo runner active'), 'teleop': _status('active' if state.active else 'unavailable', 'controller state'), 'recording': _status('recording' if phase == 'RECORDING' else 'idle', phase), 'latency_ms': None}, 'task_status': {}, 'camera': {'available': True, 'message': 'task_camera_v1 + oblique_camera_v1 + fr3_wrist_camera_v1', 'task_frame_url': '/api/work/frame?camera=task', 'oblique_frame_url': '/api/work/frame?camera=oblique', 'wrist_frame_url': '/api/work/frame?camera=wrist', 'frame_seq': time.time_ns(), 'region_overlay': overlay}, 'force': {'available': True, 'source': 'MuJoCo push-tool/block contact', 'fx': float(runner.get_contact_force_xyz()[0]), 'fy': float(runner.get_contact_force_xyz()[1]), 'fz': float(runner.get_contact_force_xyz()[2])}, 'initial_object_pose': initial_state.tolist() if initial_state is not None else None, 'initial_ee_pose': initial_ee_state.tolist() if initial_ee_state is not None else None, 'goal_pose': runner.get_goal_pose().tolist() if runner is not None else None, 'camera_config_version': 'dual_camera_v2', 'dataset_version': config.dataset_version, 'haptic': {'available': False, 'enabled': False, 'message': 'Haptic is isolated in omy_franka_teleop.'}, 'trajectory': {'available': bool(points), 'source': 'live runner EE pose', 'points': list(points), 'message': '' if points else 'No trajectory available'}, 'control_metrics': metrics, 'inventory_changed': inventory_changed})
        except (AttributeError, OSError):
            pass
    try:
        ensure_policy_import_path()
        from data_efficiency.runner import RegionRunner
        from data_efficiency.regions import REGION_VERSION, classify_region
        from projects.fr3_block_push.dataset.frame_extractor import DualCameraFrameExtractor
        from projects.fr3_block_push.dataset.recorder import DatasetRecorder, RecorderConfig
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dataset_root = shared_datasets_root() / f'{config.dataset_version}_{timestamp}_{config.region}'
        fail_root = shared_datasets_root() / f'{config.dataset_version}_{timestamp}_{config.region}_fail'
        extractor = DualCameraFrameExtractor(image_size=(96, 96))
        with RegionRunner(robot_system_root().resolve(), region=config.region, seed=config.seed, viewer_enabled=False, image_size=(96, 96), input_backend=config.input_backend, omy_endpoint=config.omy_endpoint, task=config.task) as runner:
            sample_queue: queue.Queue[Any] = queue.Queue()

            def frame_writer() -> None:
                while True:
                    sample = sample_queue.get()
                    try:
                        if sample is None:
                            return
                        frames.append(dict(extractor.extract(sample)))
                    finally:
                        sample_queue.task_done()
            writer = threading.Thread(target=frame_writer, name='work-frame-writer', daemon=True)
            writer.start()
            runner.reset()
            initial_state = runner.get_object_pose().copy()
            initial_ee_state = runner.get_ee_pose().copy()
            classified_region = classify_region(initial_state)
            if classified_region != config.region:
                raise RuntimeError(f'sampled initial object pose classified as {classified_region}, expected Region {config.region}')
            (phase, message) = ('RECORDING', 'Worker ready; one active episode buffer.')
            publish()
            next_tick = time.perf_counter()
            while runner.is_running():
                action = _drain_command(commands)
                if action:
                    if action == 'reset_initial':
                        sample_queue.join()
                        frames.clear()
                        points.clear()
                        runner.reset(block_position=initial_state)
                        (phase, message) = ('RECORDING', 'Episode reset to initial state; recording continues.')
                        publish(True)
                        continue
                    if action in {'abort', 'discard', 'reset'}:
                        (phase, message) = ('IDLE', 'Episode discarded; runtime stopped without saving.')
                        publish()
                        return
                    if action in {'success', 'stop_and_save', 'failure'}:
                        phase = 'SAVING'
                        publish()
                        sample_queue.join()
                        sample_queue.put(None)
                        writer.join(timeout=5)
                        target = _save_episode(frames, action in {'success', 'stop_and_save'}, dataset_root, fail_root, config, runner, initial_state, initial_ee_state, REGION_VERSION, DatasetRecorder, RecorderConfig)
                        dataset_root = target
                        (phase, message) = ('IDLE', 'Episode saved and validated by the existing recorder.')
                        publish(True)
                        return
                capture = runner.simulation_time % 0.05 < runner.timestep
                result = runner.step(capture_sample=capture)
                if time.perf_counter() - last_publish >= 0.08:
                    frame_dir = shared_runs_root() / 'work' / 'live'
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    for (camera_name, filename, hide_robot) in (('task_camera_v1', 'task_camera.jpg', True), ('oblique_camera_v1', 'oblique_camera.jpg', False), ('fr3_wrist_camera_v1', 'wrist_camera.jpg', False)):
                        frame = runner.render_camera(camera_name, size=448, hide_robot=hide_robot)
                        temp = frame_dir / f'{filename}.tmp'
                        from PIL import Image
                        Image.fromarray(frame, mode='RGB').save(temp, format='JPEG', quality=92, subsampling=0)
                        temp.replace(frame_dir / filename)
                if result.sample is not None:
                    sample_queue.put_nowait(result.sample)
                    (x, y, _) = extract_ee_position(runner.get_ee_pose())
                    points.append({'x': x, 'y': y})
                now = time.perf_counter()
                periods.append(now - last_tick)
                last_tick = now
                publish()
                next_tick += runner.timestep
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
    except Exception as error:
        try:
            states.put_nowait({'phase': 'ERROR', 'message': str(error), 'worker_pid': os.getpid(), 'recording': False, 'status': {'omy': _status('unavailable', 'WORK worker failed'), 'fr3': _status('unavailable', 'WORK worker failed'), 'teleop': _status('unavailable', 'WORK worker failed'), 'recording': _status('error', 'WORK worker failed'), 'latency_ms': None}, 'trajectory': {'available': bool(points), 'points': list(points)}})
        except OSError:
            pass

def _save_episode(frames: list[dict[str, Any]], success: bool, root: Path, fail_root: Path, config: WorkConfig, runner: Any, initial_object_pose: Any, initial_ee_pose: Any, region_version: str, recorder_cls: Any, recorder_config_cls: Any) -> Path:
    target = root if success else fail_root
    recorder = recorder_cls.create(recorder_config_cls(root=target, repo_id=f'local/{target.name}', dual_camera=True))
    try:
        recorder.start_episode(runner.simulation_time)
        for frame in frames:
            recorder.add_frame(frame)
        recorder.save_episode()
        with (target / 'episode_metadata.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'episode_id': 0, 'task': config.task, 'region': config.region, 'region_id': config.region, 'region_version': region_version, 'initial_object_pose': list(map(float, initial_object_pose)), 'initial_ee_pose': list(map(float, initial_ee_pose)), 'goal_pose': runner.get_goal_pose().tolist(), 'camera_config_version': 'dual_camera_v2', 'collection_seed': config.seed, 'collection_session_id': target.name, 'operator_id': config.operator_id, 'outcome': 'success' if success else 'failure', 'success': success, 'failure_type': None if success else 'FAILURE', 'dataset_version': config.dataset_version, 'episode_duration': float(runner.simulation_time), 'frame_count': len(frames), 'git_commit': _git_commit()}) + '\n')
        return target
    finally:
        recorder.finalize()

def _drain_command(commands: Any) -> str | None:
    try:
        return commands.get_nowait()
    except queue.Empty:
        return None

def _git_commit() -> str | None:
    """Capture provenance without making collection depend on git."""
    try:
        return subprocess.check_output(['git', '-C', str(policy_learning_root()), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def _status(state: str, detail: str) -> dict[str, str]:
    return {'state': state, 'detail': detail}

class InventoryCache:

    def __init__(self, ttl_s: float=15.0) -> None:
        (self.ttl_s, self._expires, self._value, self._lock) = (ttl_s, 0.0, None, threading.Lock())

    def invalidate(self) -> None:
        self._expires = 0.0

    def value(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if self._value is not None and now < self._expires:
                return self._value
            (self._value, self._expires) = (_inventory_counts(), now + self.ttl_s)
            return self._value

def _inventory_counts() -> list[dict[str, Any]]:
    counts = {key: {'key': key, 'target': None, 'success': None, 'attempts': None, 'failure': None, 'abort': None} for key in REGIONS}
    found = False
    for root in (policy_learning_root() / 'datasets', shared_datasets_root()):
        if not root.is_dir():
            continue
        for path in root.glob('**/episode_metadata.jsonl'):
            try:
                for line in path.read_text(encoding='utf-8').splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    region = str(row.get('region_id', ''))[:1].upper()
                    if region not in counts:
                        continue
                    found = True
                    item = counts[region]
                    item['attempts'] = (item['attempts'] or 0) + 1
                    if row.get('success') is True:
                        item['success'] = (item['success'] or 0) + 1
                    elif str(row.get('failure_type', '')).upper() == 'ABORT':
                        item['abort'] = (item['abort'] or 0) + 1
                    else:
                        item['failure'] = (item['failure'] or 0) + 1
            except (OSError, ValueError):
                continue
    if not found:
        return [{**value, 'available': False} for value in counts.values()]
    for value in counts.values():
        for field in ('success', 'attempts', 'failure', 'abort'):
            value[field] = value[field] or 0
        value['available'] = True
    return list(counts.values())
