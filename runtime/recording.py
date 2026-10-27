"""20 Hz recorder plus process-isolated camera/UI renderers."""
from __future__ import annotations
from dataclasses import dataclass
import multiprocessing as mp
import json
import os
from pathlib import Path
import queue
import time
from typing import Iterable
import numpy as np
from .live_runner import ControlObservation, PhysicsSnapshot
from .alignment_preview import ALIGNMENT_CAMERA, TopDownAlignmentPreview, collection_boundary_projection

@dataclass(frozen=True)
class RenderRequest:
    index: int
    physics: PhysicsSnapshot
    submitted_ns: int | None = None

@dataclass(frozen=True)
class PreviewRequest:
    physics: PhysicsSnapshot
    input_sample_time: float | None = None
    submitted_at: float | None = None
    sync_context: dict | None = None
LIVE_PREVIEW_HZ = 60

class DatasetCameraRenderer:
    """Render policy-observation cameras from the active task scene."""

    def __init__(self, model: object, *, image_size: tuple[int, int]=(480, 640), cameras: tuple[str, ...]=('operator', 'wrist'), camera_overrides: dict | None=None) -> None:
        import mujoco
        _apply_camera_overrides(model, camera_overrides)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), image_size[0])
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), image_size[1])
        self._renderers = {camera: mujoco.Renderer(model, height=image_size[0], width=image_size[1]) for camera in cameras}

    def render(self, data: object) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for (camera, renderer) in self._renderers.items():
            renderer.update_scene(data, camera=camera)
            images[camera] = _copy_image(renderer.render())
        return images

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> 'DatasetCameraRenderer':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

def _apply_camera_overrides(model: object, camera_overrides: dict | None) -> None:
    if not camera_overrides:
        return
    for (name, override) in camera_overrides.items():
        camera_id = model.camera(name).id
        if 'pos' in override:
            model.cam_pos[camera_id] = override['pos']
        if 'quat' in override:
            model.cam_quat[camera_id] = override['quat']
        if 'fovy' in override:
            model.cam_fovy[camera_id] = override['fovy']
UI_PREVIEW_CAMERAS = ('operator', 'front_operator', 'left_side')

def _copy_image(image):
    return image.copy()

def _save_image(image: np.ndarray, path: Path, image_format: str) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    options = {'compress_level': 0} if image_format == 'PNG' else {'quality': 92}
    Image.fromarray(image, mode='RGB').save(temporary, format=image_format, **options)
    temporary.replace(path)

def _camera_writer_process(xml_path: str, episode_dir: str | None, live_dir: str | None, image_size: tuple[int, int], goal_position: np.ndarray, target_body_name: str | None, camera_overrides: dict | None, requests: object, written: object, errors: object, cameras: tuple[str, ...]=('operator', 'wrist')) -> None:
    """Own all renderer/OpenGL state in a process outside the control loop."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if target_body_name is not None:
        model.body_pos[model.body(target_body_name).id] = goal_position
    _apply_camera_overrides(model, camera_overrides)
    from geomagic.teleop.runtime_profile import get_profiler
    profiler = get_profiler()
    from geomagic.teleop.sync_capture import get_sync_capture
    sync = get_sync_capture()
    if sync:
        sync.role = 'recorder'
    if profiler:
        global _save_image, _copy_image
        _copy_image = profiler.wrap('camera.image_copy', _copy_image)
        _save_image = profiler.wrap('camera.encode_write', _save_image)
    renderer: mujoco.Renderer | None = None
    try:
        renderer = mujoco.Renderer(model, height=image_size[0], width=image_size[1])
        if profiler:
            profiler.instrument(renderer, {'update_scene': 'camera.scene_update', 'render': 'camera.render'})
        profile_previous = time.monotonic()
        while True:
            request = profiler.wrap('camera.queue_wait', requests.get)() if profiler else requests.get()
            if request is None:
                return
            state = request.physics
            dequeued_ns = time.monotonic_ns() if sync else None
            data.qpos[:] = state.qpos
            data.qvel[:] = state.qvel
            if data.act.size:
                data.act[:] = state.act
            data.ctrl[:] = state.ctrl
            data.qacc_warmstart[:] = state.qacc_warmstart
            data.time = state.time
            mujoco.mj_forward(model, data)
            images: dict[str, np.ndarray] = {}
            frame_dequeued = time.monotonic()
            camera_cost = {}
            render_started_ns = time.monotonic_ns() if sync else None
            for camera in cameras:
                camera_started = time.monotonic()
                renderer.update_scene(data, camera=camera, scene_option=None)
                images[camera] = _copy_image(renderer.render())
                camera_cost[camera] = (time.monotonic() - camera_started) * 1000
                if profiler:
                    profiler.record('camera.' + camera + '.render', camera_cost[camera] / 1000)
            rendered_ns = time.monotonic_ns() if sync else None
            if episode_dir is not None:
                episode = Path(episode_dir)
                for camera in cameras:
                    image = images[camera]
                    _save_image(image, episode / 'images' / camera / f'{request.index:06d}.png', 'PNG')
            if live_dir is not None:
                live = Path(live_dir)
                _save_image(images['operator'], live / 'operator_camera.jpg', 'JPEG')
                _save_image(images['wrist'], live / 'wrist_camera.jpg', 'JPEG')
            counter_lock = written.get_lock()
            with profiler.wrap_lock('camera.counter_lock_wait', counter_lock) if profiler else counter_lock:
                written.value += 1
            if sync:
                sync.emit('dataset_write', float(data.time), frame=request.index, submitted_ns=request.submitted_ns, dequeued_ns=dequeued_ns, render_started_ns=render_started_ns, rendered_ns=rendered_ns)
            if profiler:
                now = time.monotonic()
                profiler.tick(now - profile_previous)
                profile_previous = now
    except BaseException as error:
        try:
            errors.put_nowait(f'{type(error).__name__}: {error}')
        except Exception:
            pass
        raise
    finally:
        if sync:
            sync.write()
        if profiler:
            profiler.write('.recorder.json')
        if renderer is not None:
            renderer.close()

class AsyncCameraWriter:
    """Queue state copies to an independent rendering/file-I/O process."""

    def __init__(self, xml_path: Path, *, goal_position: np.ndarray, target_body_name: str | None='target', camera_overrides: dict | None=None, episode_dir: Path | None, live_dir: Path | None=None, image_size: tuple[int, int]=(480, 640), cameras: tuple[str, ...]=('operator', 'wrist')) -> None:
        if not cameras or len(set(cameras)) != len(cameras) or (not set(cameras) <= {'operator', 'front', 'wrist'}):
            raise ValueError('Dataset cameras must be unique operator/front/wrist views')
        if live_dir is not None and (not {'operator', 'wrist'} <= set(cameras)):
            raise ValueError('Legacy live copy requires operator and wrist')
        context = mp.get_context('spawn')
        self.requests = context.Queue()
        self.errors = context.Queue()
        self._written = context.Value('i', 0)
        self.process = context.Process(target=_camera_writer_process, args=(str(xml_path), str(episode_dir) if episode_dir is not None else None, str(live_dir) if live_dir is not None else None, image_size, np.asarray(goal_position, dtype=float), target_body_name, camera_overrides, self.requests, self._written, self.errors, cameras), name='mimic-camera-writer', daemon=False)
        self.process.start()
        self.submitted = self.max_backlog = 0

    @property
    def written(self) -> int:
        return int(self._written.value)

    def metrics(self):
        return {'submitted': self.submitted, 'written': self.written, 'backlog': max(0, self.submitted - self.written), 'max_backlog': self.max_backlog}

    def submit(self, index: int, physics: PhysicsSnapshot) -> None:
        from geomagic.teleop.sync_capture import get_sync_capture
        sync = get_sync_capture()
        self.requests.put_nowait(RenderRequest(index=index, physics=physics, submitted_ns=time.monotonic_ns() if sync else None))
        self.submitted += 1
        self.max_backlog = max(self.max_backlog, self.submitted - self.written)

    def close(self, *, discard_pending: bool=False) -> None:
        if self.process.is_alive():
            if discard_pending:
                while True:
                    try:
                        self.requests.get(timeout=0.05)
                    except queue.Empty:
                        break
            self.requests.put(None)
        self.process.join(timeout=60.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
            raise RuntimeError('camera writer did not stop within 60 seconds')
        try:
            message = self.errors.get_nowait()
        except queue.Empty:
            message = None
        if self.process.exitcode not in (0, None) or message:
            raise RuntimeError(message or f'camera writer exited {self.process.exitcode}')

    def cancel(self) -> None:
        """Stop immediately when an episode is intentionally discarded."""
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        if self.process.is_alive():
            raise RuntimeError('camera writer did not stop after cancellation')

def _live_preview_process(xml_path: str, live_dir: str, image_size: tuple[int, int], operator_image_size: tuple[int, int], goal_position: np.ndarray, target_body_name: str | None, camera_overrides: dict | None, requests: object, rendered: object, first_render_ns: object, last_render_ns: object, errors: object, grasp_guidance_config: object=None, collection_world_xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None=None, cameras: tuple[str, ...]=UI_PREVIEW_CAMERAS, primary_camera: str='operator', timing_metadata: bool=False, camera_updates: object=None) -> None:
    """Render only the newest state for the display-only 60 Hz preview."""
    from concurrent.futures import ThreadPoolExecutor
    import mujoco
    from .grasp_provider import MujocoGraspProvider
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if target_body_name is not None:
        model.body_pos[model.body(target_body_name).id] = goal_position
    _apply_camera_overrides(model, camera_overrides)
    grasp_provider = MujocoGraspProvider(model, grasp_guidance_config)
    from geomagic.teleop.runtime_profile import get_profiler
    profiler = get_profiler()
    from geomagic.teleop.sync_capture import get_sync_capture
    sync = get_sync_capture()
    if sync:
        sync.role = 'preview'
    if profiler:
        global _save_image, _copy_image
        _copy_image = profiler.wrap('camera.image_copy', _copy_image)
        _save_image = profiler.wrap('camera.encode_write', _save_image)
    renderer: mujoco.Renderer | None = None
    operator_renderer: mujoco.Renderer | None = None
    image_writer: ThreadPoolExecutor | None = None
    try:
        model.vis.global_.offheight = max(model.vis.global_.offheight, image_size[0], operator_image_size[0])
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, image_size[1], operator_image_size[1])
        renderer = mujoco.Renderer(model, height=image_size[0], width=image_size[1])
        operator_renderer = mujoco.Renderer(model, height=operator_image_size[0], width=operator_image_size[1])
        if profiler:
            for camera_renderer in (renderer, operator_renderer):
                profiler.instrument(camera_renderer, {'update_scene': 'camera.scene_update', 'render': 'camera.render'})
        alignment = TopDownAlignmentPreview(model) if ALIGNMENT_CAMERA in cameras else None
        boundary_written = False
        image_writer = ThreadPoolExecutor(max_workers=3, thread_name_prefix='ui-preview-jpeg')
        period = 1.0 / LIVE_PREVIEW_HZ
        next_render = time.monotonic()
        profile_previous = next_render
        latest: PreviewRequest | None = None
        while True:
            if latest is None:
                request = profiler.wrap('camera.queue_wait', requests.get)() if profiler else requests.get()
                if request is None:
                    return
                latest = request
            while True:
                try:
                    request = requests.get_nowait()
                except queue.Empty:
                    break
                if request is None:
                    return
                latest = request
            delay = next_render - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif -delay > period:
                next_render = time.monotonic()
            state = latest.physics
            if camera_updates is not None:
                newest_override = None
                while True:
                    try:
                        newest_override = camera_updates.get_nowait()
                    except queue.Empty:
                        break
                if newest_override is not None:
                    _apply_camera_overrides(model, newest_override)
                    boundary_written = False
            timing_begin_ns = time.monotonic_ns() if timing_metadata else None
            sync_render_begin_ns = time.monotonic_ns() if sync else None
            data.qpos[:] = state.qpos
            data.qvel[:] = state.qvel
            if data.act.size:
                data.act[:] = state.act
            data.ctrl[:] = state.ctrl
            data.qacc_warmstart[:] = state.qacc_warmstart
            data.time = state.time
            mujoco.mj_forward(model, data)
            images: dict[str, np.ndarray] = {}
            live = Path(live_dir)
            if collection_world_xy_bounds is not None and (not boundary_written):
                body = collection_boundary_projection(model, data, collection_world_xy_bounds, operator_image_size, camera=primary_camera)
                boundary_path = live / 'collection_boundary.json'
                boundary_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = boundary_path.with_name(f'.{boundary_path.name}.{os.getpid()}.tmp')
                temporary.write_text(json.dumps(body, allow_nan=False))
                temporary.replace(boundary_path)
                boundary_written = True
            guidance = grasp_provider.sample(data)
            guidance['wall_time'] = time.time()
            guidance['preview_pid'] = os.getpid()
            guidance_path = live / 'grasp_guidance.json'
            guidance_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = guidance_path.with_name(f'.{guidance_path.name}.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(guidance, allow_nan=False))
            temporary.replace(guidance_path)
            writes = []
            for camera in cameras:
                if camera == ALIGNMENT_CAMERA:
                    images[camera] = alignment.render(renderer, data)
                else:
                    camera_renderer = operator_renderer if camera == primary_camera else renderer
                    camera_renderer.update_scene(data, camera=camera)
                    images[camera] = _copy_image(camera_renderer.render())
                writes.append(image_writer.submit(_save_image, images[camera], live / f'{camera}_camera.jpg', 'JPEG'))
            for write in writes:
                if profiler:
                    profiler.wrap('camera.encode_wait', write.result)()
                else:
                    write.result()
            counter_lock = rendered.get_lock()
            with profiler.wrap_lock('camera.counter_lock_wait', counter_lock) if profiler else counter_lock:
                rendered.value += 1
            render_ns = time.monotonic_ns()
            if timing_metadata:
                versions = {}
                for camera in cameras:
                    stat = (live / f'{camera}_camera.jpg').stat()
                    versions[camera] = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
                body = dict(submitted_ns=round(latest.submitted_at * 1000000000.0), render_begin_ns=timing_begin_ns, completed_ns=render_ns, render_ms=(render_ns - timing_begin_ns) / 1000000.0, submit_to_produced_ms=(render_ns / 1000000000.0 - latest.submitted_at) * 1000, sim_time=float(data.time), versions=versions)
                path = live / 'preview_timing.json'
                temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
                temporary.write_text(json.dumps(body, allow_nan=False))
                temporary.replace(path)
            if sync:
                sync.emit('preview', float(data.time), timestamp_ns=render_ns, T6_ns=render_ns, render_begin_ns=sync_render_begin_ns, submitted_ns=round(latest.submitted_at * 1000000000.0) if latest.submitted_at else None, visual_kind='jpeg_produced', **latest.sync_context or {})
            if profiler:
                render_time = render_ns / 1000000000.0
                profiler.tick(render_time - profile_previous)
                profile_previous = render_time
                if latest.submitted_at is not None:
                    profiler.record('preview.submit_to_produced', render_time - latest.submitted_at)
                if latest.input_sample_time is not None and 0 <= render_time - latest.input_sample_time < 60:
                    profiler.record('latency.input_preview_produced_approx', render_time - latest.input_sample_time)
            with first_render_ns.get_lock():
                if first_render_ns.value == 0:
                    first_render_ns.value = render_ns
            with last_render_ns.get_lock():
                last_render_ns.value = render_ns
            latest = None
            next_render += period
    except BaseException as error:
        try:
            errors.put_nowait(f'{type(error).__name__}: {error}')
        except Exception:
            pass
        raise
    finally:
        if sync:
            sync.write()
        if profiler:
            profiler.write('.preview.json')
        if image_writer is not None:
            image_writer.shutdown(wait=True)
        if operator_renderer is not None:
            operator_renderer.close()
        if renderer is not None:
            renderer.close()

class AsyncLivePreview:
    """Non-blocking, latest-only renderer for the live UI camera panes."""

    def __init__(self, xml_path: Path, *, goal_position: np.ndarray, target_body_name: str | None='target', camera_overrides: dict | None=None, live_dir: Path, image_size: tuple[int, int]=(960, 1280), operator_image_size: tuple[int, int]=(960, 1280), grasp_guidance_config: object=None, collection_world_xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None=None, cameras: tuple[str, ...]=UI_PREVIEW_CAMERAS, primary_camera: str='operator', timing_metadata: bool=False, interactive_cameras: bool=False) -> None:
        if not cameras or primary_camera not in cameras or len(set(cameras)) != len(cameras):
            raise ValueError('preview cameras must be unique and contain the primary camera')
        context = mp.get_context('spawn')
        self.requests = context.Queue(maxsize=1)
        self.errors = context.Queue()
        self.camera_updates = context.Queue() if interactive_cameras else None
        self._rendered = context.Value('i', 0)
        self._first_render_ns = context.Value('q', 0)
        self._last_render_ns = context.Value('q', 0)
        self.process = context.Process(target=_live_preview_process, args=(str(xml_path), str(live_dir), image_size, operator_image_size, np.asarray(goal_position, dtype=float), target_body_name, camera_overrides, self.requests, self._rendered, self._first_render_ns, self._last_render_ns, self.errors, grasp_guidance_config, collection_world_xy_bounds, cameras, primary_camera, timing_metadata, self.camera_updates), name='mimic-live-preview', daemon=False)
        self.process.start()
        self.submitted = self.replaced = self.skipped = 0
        self.first_submit = self.last_submit = None

    @property
    def rendered(self) -> int:
        return int(self._rendered.value)

    def set_camera_overrides(self, overrides: dict) -> None:
        """Queue a view change without restarting or touching physics."""
        if self.camera_updates is None:
            raise ValueError('Interactive preview cameras were not enabled')
        self.camera_updates.put_nowait(overrides)

    def metrics(self) -> dict[str, float | int | None]:
        count = self.rendered
        first = int(self._first_render_ns.value)
        last = int(self._last_render_ns.value)
        elapsed = (last - first) / 1000000000.0 if count > 1 and last > first else 0.0
        return {'configured_hz': LIVE_PREVIEW_HZ, 'requests_submitted': self.submitted, 'requested_fps': (self.submitted - 1) / (self.last_submit - self.first_submit) if self.submitted > 1 and self.last_submit > self.first_submit else None, 'requests_replaced': self.replaced, 'requests_skipped_full': self.skipped, 'queue_capacity': 1, 'produced_fps_only': True, 'frames': count, 'elapsed_s': elapsed, 'measured_hz': (count - 1) / elapsed if elapsed > 0.0 else None}

    def submit(self, physics: PhysicsSnapshot) -> None:
        try:
            while True:
                self.requests.get_nowait()
                self.replaced += 1
        except queue.Empty:
            pass
        from geomagic.teleop.runtime_profile import get_profiler
        profiler = get_profiler()
        from geomagic.teleop.sync_capture import get_sync_capture
        sync = get_sync_capture()
        try:
            self.requests.put_nowait(PreviewRequest(physics=physics, input_sample_time=profiler.last_qtarget_input_time if profiler else None, submitted_at=time.monotonic(), sync_context=sync.preview_context() if sync else None))
            now = time.monotonic()
            self.first_submit = now if self.first_submit is None else self.first_submit
            self.last_submit = now
            self.submitted += 1
        except queue.Full:
            self.skipped += 1

    def close(self) -> None:
        if self.process.is_alive():
            try:
                while True:
                    self.requests.get_nowait()
            except queue.Empty:
                pass
            self.requests.put(None)
        self.process.join(timeout=60.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
            raise RuntimeError('live preview did not stop within 60 seconds')
        try:
            message = self.errors.get_nowait()
        except queue.Empty:
            message = None
        if self.process.exitcode not in (0, None) or message:
            raise RuntimeError(message or f'live preview exited {self.process.exitcode}')

    def cancel(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        if self.process.is_alive():
            raise RuntimeError('live preview did not stop after cancellation')

def stack_observations(observations: Iterable[ControlObservation]) -> dict[str, np.ndarray]:
    rows = list(observations)
    if not rows:
        raise ValueError('cannot save an empty episode')
    keys = tuple(rows[0].numeric())
    return {key: np.stack([row.numeric()[key] for row in rows]) for key in keys}
