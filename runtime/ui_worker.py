"""Legacy experiment-console WORK worker for the Mimic live runner."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
import json
import csv
import os
from pathlib import Path
import queue
import shutil
import statistics
import tempfile
import time
from typing import Any
import numpy as np
from .live_runner import CONTROL_DT, CONTROL_HZ, STEPS_PER_DATASET_SAMPLE, MimicLiveRunner
from .contact_replay import replay_contact_episode, save_contact_replay, write_contact_plots
from .recording import AsyncCameraWriter, AsyncLivePreview, LIVE_PREVIEW_HZ, stack_observations
from .dataset_validation import validate_dataset
from .camera_schema import CURRENT_CAMERA_SCHEMA, LEFT_SIDE_CAMERA_CONFIG_VERSION, OPERATOR_CAMERA_CONFIG_VERSION
from .episode_naming import finalize_grouped_episode, next_attempt_sequence
from .collection_plan import accept_finalized_success, collection_plan, XYCollectionPlan
from .ui_camera_layout import FRONT_PREVIEW_CAMERAS, geomagic_square_front_panel, front_preview_overrides, front_camera_state
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_EXPERIMENTS = {'fr3-pick-place': 'pick_and_place', 'pick_and_place': 'pick_and_place', 'mimicgen_square': 'mimicgen_square', 'fr3-mimicgen-square': 'mimicgen_square'}
PHYSICAL_GEOMAGIC_CLOCK_DOMAIN = 'server.monotonic.hdapi'

def _physical_geomagic_preflight(backend: Any, device: dict[str, Any], timeout_s: float=3.0) -> dict[str, Any]:
    """Require fresh packets from this UI's own HDAPI streamer before WORK.

    Runs before the first physics step, so the episode is unaffected. A
    synthetic sender (other clock domain) or a foreign session never passes.
    """
    client = backend.client
    expected = device.get('session_id')
    deadline = time.monotonic() + timeout_s
    seen = None
    while True:
        client.poll()
        (status, state) = (client.status(), client.latest_state())
        if state is not None:
            seen = (state.session_id, state.source_clock_domain)
            if status.fresh and state.source_clock_domain == PHYSICAL_GEOMAGIC_CLOCK_DOMAIN and (state.session_id.startswith('hdapi-') or (device.get('force_output_enabled') is True and expected and state.session_id.startswith('teleop-geo-'))) and (not expected or state.session_id == expected):
                return {'passed': True, 'session_id': state.session_id, 'source_clock_domain': state.source_clock_domain, 'packets_accepted': status.packets_accepted, 'age_ms': status.age_ms}
        if time.monotonic() >= deadline:
            detail = 'no packet received on 127.0.0.1:5558' if seen is None else f"last packet session={seen[0]} domain={seen[1]}, expected physical session {expected or 'hdapi-*'} that is fresh"
            raise RuntimeError(f'physical Geomagic preflight failed: {detail}')
        time.sleep(0.002)

def _status(state: str, detail: str) -> dict[str, str]:
    return {'state': state, 'detail': detail}

def _leader_status(ui: dict[str, Any], phase: str, leader: str, source: str | None, health: dict[str, Any], ever_fresh: bool) -> dict[str, Any]:
    recording = phase == 'RECORDING'
    fresh = bool(ui['input_fresh'] and recording)
    if source == 'physical_hdapi':
        state = 'connected' if fresh else 'stale' if ever_fresh and recording else 'waiting'
        detail = 'Geomagic Touch physical HDAPI ' + {'connected': 'fresh', 'stale': 'stale (FR3 holding)', 'waiting': 'waiting for fresh input'}[state]
    else:
        state = 'connected' if fresh else 'unavailable'
        detail = ('Geomagic synthetic UDP' if leader == 'geomagic' else 'OMY input') + (' fresh' if fresh else ' waiting / stale')
    result = {**_status(state, detail), 'name': leader, 'label': 'Geomagic Touch' if leader == 'geomagic' else 'OMY-L100', 'connected': fresh, 'fresh': fresh, 'active': bool(ui['teleop_active'] and recording), 'mode': ui['teleop_mode']}
    if leader == 'geomagic':
        from geomagic.teleop.pose_mapper import MODE_LABELS
        result['mode_label'] = MODE_LABELS[ui['teleop_mode']]
        rate = health['rate_hz']
        result.update(device_source=source, stale_events=health['stale_events'], input_hz=None if rate is None else round(rate, 1))
    return result

def _next_command(commands: Any) -> str | dict[str, Any] | None:
    try:
        return commands.get_nowait()
    except queue.Empty:
        return None

def _timing(periods: deque[float], work: deque[float], misses: int) -> dict[str, Any]:
    if not periods:
        return {'available': False, 'message': 'No live control samples.'}
    values = np.asarray(periods, dtype=float)
    work_values = np.asarray(work, dtype=float)
    return {'available': True, 'configured_hz': CONTROL_HZ, 'measured_hz': float(1.0 / np.mean(values)), 'mean_period_s': float(np.mean(values)), 'p50_period_s': float(np.percentile(values, 50)), 'p95_period_s': float(np.percentile(values, 95)), 'p99_period_s': float(np.percentile(values, 99)), 'max_period_s': float(np.max(values)), 'jitter_s': float(np.std(values)), 'mean_work_s': float(np.mean(work_values)), 'deadline_misses': misses, 'samples': len(periods)}

def work_worker_main(config: dict[str, Any], commands: Any, states: Any) -> None:
    """Run control at 1 kHz; publish low-rate state snapshots to the UI."""
    if config.get('_sync_capture_directory'):
        os.environ['GEOMAGIC_SYNC_CAPTURE'] = config['_sync_capture_directory']
    region = str(config.get('region', 'A'))
    endpoint = str(config.get('omy_endpoint', 'tcp://127.0.0.1:5557'))
    seed = int(config.get('seed', 0))
    dataset_version = str(config.get('dataset_version', 'omy_fr3_mimic_20hz_diagnostic_v1'))
    operator_id = str(config.get('operator_id', 'operator_01'))
    input_backend = str(config.get('input_backend', 'zmq'))
    leader = 'geomagic' if input_backend == 'geomagic' else 'omy'
    geomagic_source = str(config.get('geomagic_source') or 'synthetic_udp') if leader == 'geomagic' else None
    physical_geomagic = geomagic_source == 'physical_hdapi'
    haptics_enabled = bool(config.get('_haptics_enabled', False))
    if haptics_enabled and (not (physical_geomagic and leader == 'geomagic')):
        raise ValueError("haptics require this UI's physical Geomagic owner")
    geomagic_device = dict(config.get('geomagic_device') or {})
    requested_task = str(config.get('task', 'fr3-pick-place'))
    experiment = TASK_EXPERIMENTS.get(requested_task, 'pick_and_place')
    front_panel = geomagic_square_front_panel(leader, geomagic_source, experiment)
    plan = collection_plan(config.get('collection_plan_id'))
    workspace_validation = config.get('workspace_validation')
    if workspace_validation:
        if not physical_geomagic or leader != 'geomagic' or experiment != 'mimicgen_square' or plan:
            raise ValueError('workspace validation requires physical Geomagic Square without a collection plan')
        from .workspace_validation import load_validation, summary
        assessment = load_validation(workspace_validation['plan_name'])
        if assessment['workspace_status'] != 'DRAFT':
            raise ValueError('validation workspace already frozen')
        if workspace_validation['point'] not in {p['name'] for p in assessment['points']}:
            raise ValueError('unknown validation point')
    if plan is not None and (plan.task != experiment or not plan.seed_start <= seed <= plan.seed_end):
        raise ValueError('collection plan/task/seed mismatch')
    if isinstance(plan, XYCollectionPlan):
        if not physical_geomagic or leader != 'geomagic':
            raise ValueError('XY50 collection requires physical Geomagic')
        plan.require_frozen()
    configured_dataset_root = Path(str(config.get('dataset_output_root'))) if config.get('dataset_output_root') else None
    if configured_dataset_root is not None and (not configured_dataset_root.is_absolute()):
        configured_dataset_root = PROJECT_ROOT / 'experiments' / configured_dataset_root
    dataset_output_root = (configured_dataset_root if configured_dataset_root is not None else PROJECT_ROOT / 'experiments' / experiment / 'datasets' / 'raw').expanduser().resolve()
    from .sample_weight import SampleWeight, save_weighted_samples
    weight = SampleWeight()
    planned_metadata = plan.episode_fields(seed, attempt_number=plan.reserve_attempt(dataset_output_root, seed)) if isinstance(plan, XYCollectionPlan) else {}
    calibration_only = os.environ.get('MIMIC_CALIBRATION_ONLY', '0').lower() in {'1', 'true', 'yes', 'on'}
    if calibration_only and leader == 'geomagic':
        raise ValueError('OMY direction calibration is unavailable for the Geomagic leader')
    calibration_output_root = None
    if calibration_only:
        dataset_version = os.environ.get('MIMIC_CALIBRATION_DATASET_VERSION', 'omy_direction_calibration_v1')
        calibration_output_root = Path(os.environ.get('MIMIC_CALIBRATION_OUTPUT_ROOT', str(PROJECT_ROOT / 'tests' / 'omy_direction_calibration' / 'recordings')))
    next_attempt_index: int | None = None if calibration_only else next_attempt_sequence(dataset_output_root)
    scratch = Path(tempfile.mkdtemp(prefix='mimic_work_', dir='/tmp'))
    configured_preview_dir = config.get('_preview_dir')
    live_dir = Path(str(configured_preview_dir)) if configured_preview_dir else PROJECT_ROOT / 'runs' / 'work' / 'live'
    preview_image_size = tuple((int(value) for value in config.get('_preview_image_size', (960, 1280))))
    if len(preview_image_size) != 2 or min(preview_image_size) <= 0:
        raise ValueError('_preview_image_size must be [height, width]')
    live_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('grasp_guidance.json', 'collection_boundary.json', 'operator_camera.jpg', 'front_camera.jpg', 'front_operator_camera.jpg', 'left_side_camera.jpg', 'top_down_camera.jpg', 'oblique_camera.jpg', 'task_camera.jpg', 'wrist_camera.jpg'):
        try:
            (live_dir / filename).unlink()
        except FileNotFoundError:
            pass
    observations = []
    high_sim_time: list[float] = []
    high_wall_time: list[float] = []
    high_actions: list[np.ndarray] = []
    high_arm_qpos: list[np.ndarray] = []
    high_ee_pose: list[np.ndarray] = []
    high_cube_pose: list[np.ndarray] = []
    high_contact: list[bool] = []
    high_diagnostics: dict[str, list[np.ndarray]] = {}
    points: deque[dict[str, float]] = deque(maxlen=600)
    periods: deque[float] = deque(maxlen=20000)
    work_times: deque[float] = deque(maxlen=20000)
    runner: MimicLiveRunner | None = None
    writer: AsyncCameraWriter | None = None
    preview: AsyncLivePreview | None = None
    preview_metrics: dict[str, Any] | None = None
    recorder_metrics = None
    phase = 'STARTING'
    message = 'Loading 1000 Hz Geomagic-FR3 Mimic runner.' if leader == 'geomagic' else 'Loading 1000 Hz OMY-FR3 Mimic runner.'
    outcome: str | None = None
    deadline_misses = 0
    from geomagic.teleop.runtime_profile import get_profiler
    profile = get_profiler() if leader == 'geomagic' else None
    compute_timing = profile.wrap('ui.percentile_stats', _timing) if profile else _timing
    measure_only = bool(workspace_validation) or (leader == 'geomagic' and os.environ.get('MIMIC_GEOMAGIC_MEASURE_ONLY') == '1')
    detailed_logs = leader != 'geomagic' or profile is not None or os.environ.get('MIMIC_GEOMAGIC_DIAGNOSTICS_MODE') == 'detailed'
    timing_cache = {}
    timing_cache_at = -float('inf')
    phase_rates = {name: {'physics_steps': 0, 'control_updates': 0, 'wall_s': 0.0} for name in ('idle', 'position', 'rotation')}
    previous_updates = 0

    def rate_summary():
        return {name: dict(value, physics_wall_hz=value['physics_steps'] / value['wall_s'] if value['wall_s'] else 0.0, dls_wall_hz=value['control_updates'] / value['wall_s'] if value['wall_s'] else 0.0, rtf=value['physics_steps'] * 0.001 / value['wall_s'] if value['wall_s'] else 0.0) for (name, value) in phase_rates.items()}
    ever_input_fresh = False
    input_health: dict[str, Any] = {'previous_fresh': False, 'stale_events': 0, 'max_age_ms': None, 'rate_t': None, 'rate_n': None, 'rate_hz': None, 'start_t': None, 'start_n': None, 'preflight': None}
    initial_robot_table_contact: bool | None = None
    started = time.perf_counter()
    last_publish = 0.0

    def publish(*, force: bool=False, dataset_root: str | None=None) -> None:
        nonlocal last_publish, timing_cache, timing_cache_at
        now = time.perf_counter()
        if not force and now - last_publish < 1.0 / LIVE_PREVIEW_HZ:
            return
        last_publish = now
        if force or now - timing_cache_at >= 1.0:
            timing_cache = compute_timing(periods, work_times, deadline_misses)
            timing_cache['phase_rates'] = rate_summary()
            timing_cache_at = now
        if runner is None:
            payload = {'phase': phase, 'message': message, 'recording': False, 'worker_pid': os.getpid(), 'trajectory': {'available': False, 'points': []}}
        else:
            ui = runner.state_for_ui()
            initial = runner.object_pose_in(runner.initial_state.qpos)
            initial_ee = runner.initial_ee_pose
            camera_ready = preview is not None and preview.rendered > 0
            payload = {'phase': phase, 'message': message, 'active_region': region, 'task': experiment, 'dataset_root': dataset_root, 'dataset_output_root': str(dataset_output_root), 'dataset_validation': saved_validation, 'next_attempt_index': next_attempt_index, 'collection_attempt': planned_metadata or None, 'episode': 0, 'frame': len(observations), 'elapsed_s': now - started, 'recording': phase == 'RECORDING' and (not measure_only), 'weight': weight.snapshot(), 'teleop_mode': ui['teleop_mode'], 'mode_change_events': ui['mode_change_events'], 'worker_pid': os.getpid(), 'status': {'leader': _leader_status(ui, phase, leader, geomagic_source, input_health, ever_input_fresh), 'omy': _status('unavailable', 'OMY is not selected') if leader == 'geomagic' else _status('connected' if ui['input_fresh'] else 'unavailable', 'fresh OMY IPC state' if ui['input_fresh'] else 'OMY input absent or stale'), 'fr3': _status('connected', 'FR3 motion frozen for OMY-only calibration' if calibration_only else 'Mimic MuJoCo model at dt=0.001 s'), 'teleop': _status('active' if ui['teleop_active'] else 'available', 'clutch engaged' if ui['teleop_active'] else 'waiting for clutch'), 'recording': _status('idle' if measure_only else 'recording' if phase == 'RECORDING' else 'idle', 'Runtime measurement; recording disabled' if measure_only else phase), 'latency_ms': None}, 'task_status': {'reason': outcome, 'success': ui['success'], 'terminated': phase == 'IDLE', 'block_goal_error_xy': ui['goal_error'], 'initial_robot_table_contact': initial_robot_table_contact}, 'camera': {'available': camera_ready, 'message': 'operator + front_operator + left_side UI views', 'operator_frame_url': '/api/work/frame?camera=operator', 'front_operator_frame_url': '/api/work/frame?camera=front_operator', 'left_side_frame_url': '/api/work/frame?camera=left_side', 'frame_seq': time.time_ns(), 'preview_hz': LIVE_PREVIEW_HZ, 'preview_measured_hz': preview.metrics().get('measured_hz') if preview is not None else None, **(front_camera_state() if front_panel else {})}, 'force': {'available': False, 'source': 'not exposed by this task runtime', 'fx': None, 'fy': None, 'fz': None}, 'initial_object_pose': initial.tolist(), 'initial_ee_pose': initial_ee.tolist(), 'goal_pose': ui['goal_position'], 'camera_config_version': CURRENT_CAMERA_SCHEMA, 'operator_camera_config_version': OPERATOR_CAMERA_CONFIG_VERSION, 'left_side_camera_config_version': LEFT_SIDE_CAMERA_CONFIG_VERSION, 'dataset_version': dataset_version, 'calibration_only': calibration_only, 'haptic': runner.haptics.snapshot() if runner.haptics is not None else {'available': False, 'enabled': False, 'message': 'OFF' if leader == 'geomagic' else 'Haptic is intentionally excluded.'}, 'trajectory': {'available': bool(points), 'source': 'live FR3 EE', 'points': list(points), 'message': '' if points else 'No trajectory yet'}, 'control_metrics': timing_cache, 'inventory_changed': phase == 'IDLE' and dataset_root is not None}
        try:
            states.put_nowait(payload)
        except queue.Full:
            try:
                states.get_nowait()
            except queue.Empty:
                pass
            try:
                states.put_nowait(payload)
            except queue.Full:
                pass
        except OSError:
            pass
    if profile:
        publish = profile.wrap('ui.status_construct_publish', publish)
        profile.instrument(states, {'put_nowait': 'ui.ipc_put_nowait'})
        profile.instrument(commands, {'get_nowait': 'scheduler.command_queue_get'})
        from multiprocessing.reduction import ForkingPickler
        ForkingPickler.dumps = staticmethod(profile.wrap('ui.ipc_serialization', ForkingPickler.dumps))
    saved_root: Path | None = None
    saved_validation: dict[str, Any] | None = None
    haptic_sender = None
    try:
        if haptics_enabled:
            from geomagic.standalone.device import ForceSender
            haptic_sender = ForceSender(config['_haptic_socket'], config['_haptic_token'])
        from .touch_weight_collection import enabled as weight_collection_enabled, TRAIN_DISTRIBUTION
        weight_collection = weight_collection_enabled()
        if weight_collection and (not physical_geomagic or experiment != 'mimicgen_square' or plan is not None):
            raise ValueError('Named weight collection requires physical Touch Square without a canonical output plan')
        runner = MimicLiveRunner(input_backend=input_backend, omy_endpoint=endpoint, seed=seed, distribution_id=f"workspace_validation:{workspace_validation['plan_name']}:{workspace_validation['point']}" if workspace_validation else plan.distribution_id if plan else 'square_restricted30_v1' if weight_collection else None, freeze_fr3_motion=calibration_only, task=requested_task, haptic_sender=haptic_sender)
        weight_collection_provenance = None
        if weight_collection:
            from .touch_weight_collection import apply_research_initial_state
            weight_collection_provenance = apply_research_initial_state(runner, seed)
        if profile:
            profile.instrument(runner.env, {'control': 'simulation.env_control', 'step_physics': 'simulation.mujoco_step'})
            profile.instrument(runner, {'post_step_state': 'recorder.state_copy', 'capture_physics': 'recorder.physics_snapshot', 'observe': 'recorder.sample_20hz', 'state_for_ui': 'ui.status_state'})
        diagnostics = runner.backend.diagnostic_state(detailed=detailed_logs) if leader == 'geomagic' else runner.backend.diagnostic_state()
        high_diagnostics = {key: [] for key in diagnostics}
        if physical_geomagic:
            message = 'Verifying fresh physical Geomagic Touch packets.'
            publish(force=True)
            input_health['preflight'] = _physical_geomagic_preflight(runner.backend, geomagic_device)
        if leader == 'geomagic':
            accepted = runner.backend.client.status().packets_accepted
            input_health.update(start_t=time.perf_counter(), start_n=accepted, rate_t=time.perf_counter(), rate_n=accepted)
        initial_robot_table_contact = runner.robot_table_contact()
        if initial_robot_table_contact:
            raise RuntimeError('collection preflight failed: initial robot-table contact')
        preview = AsyncLivePreview(runner.xml_path, goal_position=runner.initial_goal, target_body_name=runner.target_body_name, camera_overrides=front_preview_overrides(runner.camera_overrides) if front_panel else runner.camera_overrides, live_dir=live_dir, image_size=preview_image_size, grasp_guidance_config=getattr(runner.env, 'grasp_guidance_config', None), collection_world_xy_bounds=plan.world_xy_bounds if plan else None, **dict(cameras=FRONT_PREVIEW_CAMERAS, primary_camera='operator') if front_panel else {})
        writer = None if measure_only else AsyncCameraWriter(runner.xml_path, goal_position=runner.initial_goal, target_body_name=runner.target_body_name, camera_overrides=runner.camera_overrides, episode_dir=scratch, live_dir=None)
        if getattr(runner.backend, 'sync', None):
            runner.backend.sync.settings['recording_enabled'] = writer is not None
        if profile:
            profile.instrument(preview, {'submit': 'camera.preview_submit'})
            if writer is not None:
                profile.instrument(writer, {'submit': 'camera.recorder_submit'})
        weight.reset()
        phase = 'RECORDING'
        message = '1 kHz physics; Geomagic control 500 Hz simulation time; dataset 20 Hz.' if leader == 'geomagic' else '1 kHz control active; dataset 20 Hz; live preview 60 Hz.'
        if measure_only:
            message = 'Runtime measurement active; recording disabled; live preview enabled.'
        publish(force=True)
        step = 0
        next_preview_wall = time.perf_counter()
        previous_tick = time.perf_counter()
        next_deadline = previous_tick + CONTROL_DT
        catchup_steps = 20 if leader == 'geomagic' else 5
        while True:
            command = _next_command(commands)
            if isinstance(command, dict):
                if command.get('command') == 'weight':
                    if not measure_only:
                        weight.set(command.get('weight'), command.get('recording_id'), command.get('request_id'))
                else:
                    raise ValueError('unsupported structured collection command')
                publish(force=True)
                command = None
            if command is not None and command.startswith('teleop_mode:'):
                requested_mode = command.partition(':')[2]
                changed = runner.set_teleop_mode(requested_mode)
                message = f'Teleop mode changed to {requested_mode.upper()}.' if changed else f'Teleop mode remains {requested_mode.upper()}.'
                publish(force=True)
                command = None
            if command is not None and command.startswith('gripper:'):
                setter = getattr(runner.backend, 'set_gripper_command', None)
                try:
                    value = float(command.partition(':')[2])
                    if not np.isfinite(value) or not 0 <= value <= 255:
                        raise ValueError('expected a finite value in [0, 255]')
                    if setter is None:
                        message = 'This backend receives gripper commands from its existing input source.'
                    else:
                        setter(value)
                        message = f'Gripper command set to {value:g}.'
                except ValueError as error:
                    message = f'Invalid gripper command: {error}'
                publish(force=True)
                command = None
            if command in {'abort', 'discard', 'reset'}:
                outcome = 'abort' if command == 'abort' else 'discard'
                phase = 'IDLE'
                message = 'Episode discarded; worker stopped.'
                break
            if command == 'reset_initial':
                runner.reset_to_initial()
                observations.clear()
                weight.reset()
                high_sim_time.clear()
                high_wall_time.clear()
                high_actions.clear()
                high_arm_qpos.clear()
                high_ee_pose.clear()
                high_cube_pose.clear()
                high_contact.clear()
                for values in high_diagnostics.values():
                    values.clear()
                points.clear()
                step = 0
                message = 'Episode reset to its exact initial MuJoCo state.'
                publish(force=True)
                continue
            if command in {'success', 'failure', 'stop_and_save'}:
                if command == 'failure':
                    outcome = 'failure'
                elif command == 'success':
                    outcome = 'success'
                else:
                    outcome = ('success' if runner.env._is_success() else 'failure') if weight_collection else 'diagnostic'
                if measure_only:
                    outcome = 'abort'
                phase = 'SAVING'
                message = 'Finishing asynchronous camera writes.'
                publish(force=True)
                break
            work_started = time.perf_counter()
            capture = not measure_only and step % STEPS_PER_DATASET_SAMPLE == 0
            (action, observation) = runner.step_once(capture_observation=capture, before_step=(lambda observation: weight.append(len(observations))) if capture else None)
            preview_now = time.perf_counter()
            if preview is not None and preview_now >= next_preview_wall:
                preview.submit(runner.capture_physics())
                next_preview_wall = preview_now + 1.0 / LIVE_PREVIEW_HZ
            if not measure_only:
                state = runner.post_step_state(detailed_diagnostics=detailed_logs) if leader == 'geomagic' else runner.post_step_state()
                high_sim_time.append(float(runner.data.time))
                high_wall_time.append(time.perf_counter() - started)
                high_actions.append(action.copy())
                high_arm_qpos.append(state['arm_qpos'])
                high_ee_pose.append(state['ee_pose'])
                high_cube_pose.append(state['object_pose'])
                high_contact.append(bool(state['contact']))
                for (key, values) in high_diagnostics.items():
                    values.append(state[key])
            ever_input_fresh = ever_input_fresh or runner.input_fresh
            if leader == 'geomagic':
                fresh = runner.input_fresh
                if input_health['previous_fresh'] and (not fresh):
                    input_health['stale_events'] += 1
                input_health['previous_fresh'] = fresh
                if ever_input_fresh and step % 10 == 0:
                    age = runner.backend.client.status().age_ms
                    if age is not None and (input_health['max_age_ms'] is None or age > input_health['max_age_ms']):
                        input_health['max_age_ms'] = age
                if step % 500 == 0:
                    rate_now = time.perf_counter()
                    accepted = runner.backend.client.status().packets_accepted
                    if rate_now > input_health['rate_t']:
                        input_health['rate_hz'] = (accepted - input_health['rate_n']) / (rate_now - input_health['rate_t'])
                    input_health.update(rate_t=rate_now, rate_n=accepted)
            if observation is not None:
                observations.append(observation)
                writer.submit(len(observations) - 1, observation.physics)
                ee = observation.ee_pose[4:]
                points.append({'x': float(ee[0]), 'y': float(ee[1])})
            work_finished = time.perf_counter()
            work_times.append(work_finished - work_started)
            remaining = next_deadline - work_finished
            if remaining > 0.0:
                sleep_started = time.perf_counter() if profile else None
                time.sleep(remaining)
                if profile:
                    profile.record('scheduler.sleep', time.perf_counter() - sleep_started)
            else:
                deadline_misses += 1
            now = time.perf_counter()
            if leader == 'geomagic':
                phase_name = ('position' if runner.backend.pipeline.mapper.mode == 'position' else 'rotation') if runner.teleop_active else 'idle'
                rate = phase_rates[phase_name]
                rate['physics_steps'] += 1
                rate['wall_s'] += now - previous_tick
                rate['control_updates'] += max(0, runner.backend.control_updates - previous_updates)
                previous_updates = runner.backend.control_updates
            if profile:
                profile.tick(now - previous_tick)
            periods.append(now - previous_tick)
            previous_tick = now
            next_deadline += CONTROL_DT
            if now - next_deadline > catchup_steps * CONTROL_DT:
                next_deadline = now + CONTROL_DT
            step += 1
            publish()
        runner.stop_haptics()
        sync = getattr(runner.backend, 'sync', None)
        if sync:
            sync.write()
        preview_metrics = preview.metrics()
        recorder_metrics = writer.metrics() if writer is not None else None
        if outcome in {'abort', 'discard'}:
            if writer is not None:
                writer.close(discard_pending=True) if sync else writer.cancel()
            if profile or sync:
                preview.close()
                preview_metrics = preview.metrics()
            else:
                preview.cancel()
        else:
            writer.close() if writer is not None else None
            preview.close()
            preview_metrics = preview.metrics()
        writer = None
        preview = None
        if outcome in {'success', 'failure', 'diagnostic'} and observations:
            samples_20hz = stack_observations(observations)
            save_weighted_samples(scratch, samples_20hz, weight)
            initial = runner.initial_state
            np.savez(scratch / 'initial_state.npz', qpos=initial.qpos, qvel=initial.qvel, act=initial.act, ctrl=initial.ctrl, qacc_warmstart=initial.qacc_warmstart, time=np.asarray(initial.time, dtype=np.float64), goal_position=runner.initial_goal)
            original_arm = np.stack(high_arm_qpos)
            original_ee = np.stack(high_ee_pose)
            original_cube = np.stack(high_cube_pose)
            original_contact = np.asarray(high_contact, dtype=np.bool_)
            np.savez(scratch / 'original_1000hz.npz', sim_time=np.asarray(high_sim_time, dtype=np.float64), wall_time=np.asarray(high_wall_time, dtype=np.float64), action=np.stack(high_actions), qtarget=np.stack(high_actions)[:, :7], arm_qpos=original_arm, ee_pose=original_ee, cube_pose=original_cube, contact=original_contact, **{key: np.stack(values) for (key, values) in high_diagnostics.items()})
            recorded_input_metrics = runner.input_metrics() if leader == 'geomagic' else None
            (replay_metrics, replay) = replay_contact_episode(runner, actions_20hz=samples_20hz['action'], original_arm_qpos=original_arm, original_ee_pose=original_ee, original_cube_pose=original_cube, original_contact=original_contact)
            save_contact_replay(scratch, replay_metrics, replay)
            write_contact_plots(scratch, original_arm_qpos=original_arm, original_ee_pose=original_ee, original_cube_pose=original_cube, original_contact=original_contact, replay=replay)
            sim_duration = float(high_sim_time[-1] - initial.time)
            wall_duration = float(high_wall_time[-1] - high_wall_time[0] + CONTROL_DT)
            sample_times = np.asarray(samples_20hz['sim_time'], dtype=float)
            measured_dataset_hz = float(1.0 / np.mean(np.diff(sample_times))) if sample_times.size > 1 else None
            input_metrics = recorded_input_metrics if leader == 'geomagic' else runner.input_metrics()
            initial_cube_pose = runner.object_pose_in(initial.qpos)
            save_moment = time.localtime()
            full_timestamp = time.strftime('%Y%m%d_%H%M%S', save_moment)
            metadata = {'task': experiment, 'experiment': experiment, 'environment': runner.task, 'initial_state_distribution': plan.distribution_id if plan else TRAIN_DISTRIBUTION if weight_collection else 'square_uniform_eval_v1' if experiment == 'mimicgen_square' else None, 'distribution_batch_size': plan.target if plan else 50 if experiment == 'mimicgen_square' else None, 'distribution_seed': seed, 'collection_plan_id': plan.distribution_id if plan else None, 'collection_accepted': False if plan else None, 'collection_plan_config': {'seed_start': plan.seed_start, 'seed_end': plan.seed_end, 'advance_condition': plan.advance_condition, 'world_xy_bounds': plan.world_xy_bounds, 'yaw_center': plan.yaw_center, 'yaw_half_range': plan.yaw_half_range} if plan else None, 'initial_canonical_yaw_offset': runner.env.reset_distribution.yaw_offset(runner.env.canonical_reset_yaw) if plan else None, 'initial_object_yaw': float(2.0 * np.arctan2(initial_cube_pose[6], initial_cube_pose[3]) % (2.0 * np.pi)), 'initial_canonical_yaw': float(runner.env.canonical_reset_yaw if plan else (2.0 * np.arctan2(initial_cube_pose[6], initial_cube_pose[3]) - np.pi / 2.0) % (2.0 * np.pi)) if experiment == 'mimicgen_square' else None, 'collection_seed': seed, 'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S', save_moment), 'attempt_index': None, 'camera_schema': CURRENT_CAMERA_SCHEMA, 'camera_config_version': CURRENT_CAMERA_SCHEMA, 'operator_camera_config_version': OPERATOR_CAMERA_CONFIG_VERSION, 'left_side_camera_config_version': LEFT_SIDE_CAMERA_CONFIG_VERSION, 'region_label': region, 'operator_id': operator_id, 'outcome': outcome, 'hardware_source': input_backend, 'hardware_verified_by_this_run': bool(input_backend in {'ros', 'zmq'} and ever_input_fresh), 'control_hz': CONTROL_HZ, 'dataset_hz': 20, 'measured_dataset_hz': measured_dataset_hz, 'dataset_samples': len(observations), 'high_rate_samples': len(high_actions), 'simulated_duration_s': sim_duration, 'wall_duration_s': wall_duration, 'real_time_factor': sim_duration / wall_duration, 'temporal_convention': 'observation_t_before_step -> action_t', 'action_shape': [8], 'haptic': haptics_enabled, 'control_metrics': _timing(periods, work_times, deadline_misses), 'ui_preview_metrics': preview_metrics, 'teleop_command_update_rate_hz': len(high_actions) / wall_duration, 'input_metrics': input_metrics, 'initial_object_pose': initial_cube_pose.tolist(), 'final_object_pose': original_cube[-1].tolist(), 'object_displacement_m': float(np.linalg.norm(original_cube[-1, :3] - initial_cube_pose[:3])), 'initial_cube_pose': initial_cube_pose.tolist(), 'final_cube_pose': original_cube[-1].tolist(), 'cube_displacement_m': float(np.linalg.norm(original_cube[-1, :3] - initial_cube_pose[:3])), 'contact_samples': int(np.count_nonzero(original_contact)), 'contact_present': bool(np.any(original_contact)), 'contact_replay_pilot_pass': replay_metrics['pilot_pass'], 'clutch_engagements': int(high_diagnostics['clutch_count'][-1]), 'teleop_mode': str(np.asarray(high_diagnostics['teleop_mode'][-1]).item()), 'mode_change_events': list(runner.backend.mode_change_events), 'calibration_only': calibration_only, 'fr3_motion_frozen': calibration_only}
            metadata.update(planned_metadata)
            metadata['sample_weight'] = weight.metadata()
            if weight_collection:
                metadata.update(weight_collection='manual_touch_square', research_collection=True, production=False, test_only=False, source_training_distribution=TRAIN_DISTRIBUTION, source_training_state_index=seed % 50, weight_collection_provenance=weight_collection_provenance, collection_run_id=config.get('collection_run_id'), manual_collection=True)
            if leader == 'geomagic':
                from geomagic.standalone.ui_haptics import haptic_metadata
                metadata.update(haptic_metadata(haptics_enabled))
                client = runner.backend.client
                final = client.latest_state() if client is not None else None
                final_status = client.status() if client is not None else None
                elapsed = time.perf_counter() - input_health['start_t'] if input_health['start_t'] is not None else None
                accepted_hz = (final_status.packets_accepted - input_health['start_n']) / elapsed if final_status is not None and elapsed else None
                preflight = input_health['preflight']
                physical_verified = bool(physical_geomagic and preflight and preflight.get('passed') and ever_input_fresh and (final is not None) and (final.source_clock_domain == PHYSICAL_GEOMAGIC_CLOCK_DOMAIN) and (final.session_id.startswith('hdapi-') or (haptics_enabled and final.session_id == geomagic_device.get('session_id') and final.session_id.startswith('teleop-geo-'))))
                metadata.update({'leader_device': 'geomagic_touch', 'dataset_version': dataset_version, 'input_source': geomagic_source, 'synthetic_input': not physical_geomagic, 'hardware_verified_by_this_run': physical_verified, 'geomagic_input': {'device_source': geomagic_source, 'device': geomagic_device or None, 'preflight': preflight, 'final_session_id': final.session_id if final is not None else None, 'final_source_clock_domain': final.source_clock_domain if final is not None else None, 'session_changes': final_status.session_changes if final_status else None, 'accepted_input_hz': accepted_hz, 'stale_events': input_health['stale_events'], 'max_input_age_ms': input_health['max_age_ms']}})
            if calibration_only or outcome not in {'success', 'failure'}:
                (scratch / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
                save_parent = calibration_output_root if calibration_output_root is not None else dataset_output_root
                save_parent.mkdir(parents=True, exist_ok=True)
                saved_root = save_parent / f'{dataset_version}_{full_timestamp}_{outcome}'
                if saved_root.exists():
                    raise FileExistsError(f'refusing to overwrite recording: {saved_root}')
                shutil.move(str(scratch), str(saved_root))
            else:
                dataset_output_root.mkdir(parents=True, exist_ok=True)
                (saved_root, _attempt_index) = finalize_grouped_episode(scratch, dataset_output_root, outcome, metadata, hhmm=time.strftime('%H%M', save_moment))
            saved_validation = validate_dataset(saved_root)
            (saved_root / 'validation.json').write_text(json.dumps(saved_validation, indent=2) + '\n', encoding='utf-8')
            if plan and outcome == 'success':
                accept_finalized_success(saved_root, plan, saved_validation)
            if calibration_only:
                _write_calibration_tables(saved_root, high_diagnostics, high_sim_time)
            phase = 'IDLE'
            replay_result = 'PASS' if replay_metrics['pilot_pass'] else 'FAIL'
            message = f'Contact episode and 20 Hz ZOH replay saved; pilot validation {replay_result} (not yet LeRobotDataset).'
        publish(force=True, dataset_root=str(saved_root) if saved_root else None)
    except BaseException as error:
        phase = 'ERROR'
        message = f'{type(error).__name__}: {error}'
        publish(force=True)
    finally:
        if config.get('collection_run_id'):
            from .collection_counts import event as collection_event
            try:
                collection_event(dataset_output_root, config['collection_run_id'], 'finished', phase=phase, saved_root=str(saved_root) if saved_root else None)
            except Exception as error:
                print(f'Collection count update failed: {error}', flush=True)
        if haptic_sender is not None:
            haptic_sender.close()
        if runner is not None:
            sync = getattr(runner.backend, 'sync', None)
            if sync:
                try:
                    sync.write()
                except Exception:
                    pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if preview is not None:
            try:
                preview.close()
            except Exception:
                pass
        if runner is not None:
            report_path = os.environ.get('MIMIC_GEOMAGIC_RUNTIME_REPORT') if leader == 'geomagic' else None
            if report_path:
                destination = Path(report_path).expanduser()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({'phase_rates': rate_summary(), 'preview': preview_metrics, 'input': runner.input_metrics(), 'recorder': recorder_metrics, 'recording_enabled': not measure_only, 'diagnostics_mode': os.environ.get('MIMIC_GEOMAGIC_DIAGNOSTICS_MODE', 'legacy')}, indent=2, allow_nan=False) + '\n')
            runner.close()
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

def _write_calibration_tables(root: Path, diagnostics: dict[str, list[np.ndarray]], sim_times: list[float]) -> None:
    """Write OMY-only samples and automatically detected clutch trials."""
    fields = ['sim_time', 'input_fresh', 'teleop_active', 'clutch_count', 'trigger_position', *[f'omy_joint_{i}' for i in range(1, 7)], 'omy_qw', 'omy_qx', 'omy_qy', 'omy_qz', 'omy_x', 'omy_y', 'omy_z']
    with (root / 'omy_fk_samples.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for (i, sim_time) in enumerate(sim_times):
            joints = np.asarray(diagnostics['omy_joint_position'][i]).reshape(-1)
            pose = np.asarray(diagnostics['omy_ee_pose'][i]).reshape(-1)
            writer.writerow([sim_time, int(np.asarray(diagnostics['input_fresh'][i]).item()), int(np.asarray(diagnostics['teleop_active'][i]).item()), int(np.asarray(diagnostics['clutch_count'][i]).item()), float(np.asarray(diagnostics['trigger_position'][i]).item()), *map(float, joints), *map(float, pose)])
    active = np.asarray([bool(np.asarray(value).item()) for value in diagnostics['teleop_active']])
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    expected_labels = ['Forward'] * 3 + ['Backward'] * 3 + ['Left'] * 3 + ['Right'] * 3 + ['Up'] * 3 + ['Down'] * 3
    trials = []
    for (trial_index, (start, stop)) in enumerate(zip(starts, stops), start=1):
        initial = np.asarray(diagnostics['omy_ee_pose'][start])[4:7]
        final = np.asarray(diagnostics['omy_ee_pose'][stop])[4:7]
        delta = final - initial
        dominant = int(np.argmax(np.abs(delta)))
        trials.append({'trial_index': trial_index, 'clutch_count': int(np.asarray(diagnostics['clutch_count'][start]).item()), 'direction_label': expected_labels[trial_index - 1] if trial_index <= len(expected_labels) else '', 'start_s': float(sim_times[start]), 'stop_s': float(sim_times[stop]), 'initial_xyz_m': initial.tolist(), 'final_xyz_m': final.tolist(), 'delta_xyz_m': delta.tolist(), 'dominant_axis': 'XYZ'[dominant], 'dominant_sign': '+' if delta[dominant] >= 0 else '-'})
    (root / 'omy_fk_trials.json').write_text(json.dumps({'source': 'OMY FK only; FR3 motion frozen', 'label_order_expected': ['Forward', 'Backward', 'Left', 'Right', 'Up', 'Down'], 'trials': trials}, indent=2) + '\n', encoding='utf-8')
