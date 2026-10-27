"""Small stdlib HTTP API for the experiment console.

Run from the workspace root with:
    python3 -m experiment_console.backend.server
"""
from __future__ import annotations
import argparse
import atexit
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import signal
import threading
import time
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from .analyze_adapter import AnalyzeAdapter
from .eval_adapter import EvalAdapter
from .train_adapter import TrainAdapter
from .work_adapter import WorkAdapter, WorkConfig
from .experiment_paths import create_folder, folder_listing, inspect_path
from .conversion_adapter import ConversionAdapter
from .native_folder_picker import NativeFolderPicker
from .ui_session import UISessions

class ConsoleService:

    def __init__(self) -> None:
        self.work = WorkAdapter()
        self.train = TrainAdapter()
        self.eval = EvalAdapter()
        self.analyze = AnalyzeAdapter()
        self.conversion = ConversionAdapter()
        self.folder_picker = NativeFolderPicker()
        self._close_lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    def state(self, view: str) -> dict:
        work = self.work.state()
        response = {'source': 'live', 'view': view, 'header': _header(work), 'work': work}
        if view == 'train':
            response['train'] = self.train.state()
            response['conversion'] = self.conversion.state()
        elif view == 'eval':
            response['eval'] = self.eval.state()
        elif view == 'analyze':
            response['analyze'] = self.analyze.snapshot()
        elif view != 'work':
            raise ValueError(f'unknown view: {view}')
        return response

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        for component in (self.folder_picker, self.work, self.train, self.eval, self.conversion):
            try:
                component.close()
            except Exception as error:
                print(f'Console cleanup: {type(component).__name__}: {error}', flush=True)
SERVICE = ConsoleService()
UI_SESSIONS = UISessions()
SHUTDOWN_REQUESTED = threading.Event()
SERVER = None

def _encode_json(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False).encode('utf-8')

def _enable_runtime_profile() -> None:
    """Opt-in HTTP-process measurements, separate from the physics process."""
    if os.environ.get('MIMIC_GEOMAGIC_PROFILE') != '1':
        return
    from geomagic.teleop.runtime_profile import get_profiler
    profiler = get_profiler()
    global _encode_json
    _encode_json = profiler.wrap('frontend.json_serialization', _encode_json)
    profiler.instrument(Handler, {'_json': 'frontend.http_json_publish', '_binary': 'frontend.http_frame_publish'})
    profiler.instrument(SERVICE, {'state': 'frontend.status_construct'})
    runtime = SERVICE.work.runtime
    profiler.instrument(runtime, {'_drain_states': 'frontend.ipc_receive'})
    runtime._lock = profiler.wrap_lock('frontend.runtime_lock_wait', runtime._lock)
    started = time.monotonic()

    def write_frontend_report():
        profiler.wall['idle'] = time.monotonic() - started
        profiler.write('.frontend.json')
    atexit.register(write_frontend_report)

def _header(work: dict) -> dict:
    status = work['status']
    return {'omy': status['omy'], 'leader': status.get('leader'), 'fr3': status['fr3'], 'teleop': status['teleop'], 'recording': status['recording'], 'latency_ms': status.get('latency_ms'), 'session': work.get('dataset_root'), 'task': work.get('task'), 'region': work.get('active_region')}

class Handler(BaseHTTPRequestHandler):
    server_version = 'ExperimentConsole/0.1'

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == '/':
                body = {'service': 'mimic experiment console backend', 'ok': True, 'frontend': 'http://127.0.0.1:5173', 'health': '/api/health', 'state': '/api/state?mode=live&view=work'}
            elif parsed.path == '/api/health':
                body = {'ok': True, 'source': 'live'}
            elif parsed.path == '/api/state':
                mode = query.get('mode', ['live'])[0]
                if mode != 'live':
                    raise ValueError('mock data is frontend-local and must be explicitly selected')
                body = SERVICE.state(query.get('view', ['work'])[0])
            elif parsed.path == '/api/mujoco/state':
                body = SERVICE.work.runtime.mujoco_state()
            elif parsed.path == '/api/train/state':
                body = SERVICE.train.state()
            elif parsed.path == '/api/conversion/state':
                body = SERVICE.conversion.state()
            elif parsed.path == '/api/datasets':
                body = {'datasets': SERVICE.train.datasets()}
            elif parsed.path == '/api/experiment/folders':
                body = folder_listing(query.get('path', [None])[0])
            elif parsed.path == '/api/experiment/validate':
                body = inspect_path(query.get('path', [None])[0])
            elif parsed.path == '/api/eval/state':
                body = SERVICE.eval.state()
            elif parsed.path == '/api/eval/frame':
                frame = SERVICE.eval.frame_bytes()
                if frame is None:
                    self._json(404, {'error': 'no evaluation frame available'})
                    return
                self._binary(200, frame, 'image/jpeg')
                return
            elif parsed.path == '/api/work/frame':
                frame = SERVICE.work.frame_bytes(query.get('camera', ['operator'])[0])
                if frame is None:
                    self._json(404, {'error': 'no work camera frame available'})
                    return
                self._binary(200, frame, 'image/jpeg')
                return
            elif parsed.path == '/api/analyze/state':
                body = SERVICE.analyze.snapshot()
            else:
                self._json(404, {'error': 'not found'})
                return
            self._json(200, body)
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            self._json(400, {'error': str(error)})

    def do_POST(self) -> None:
        try:
            payload = self._payload()
            if self.path == '/api/mujoco/start':
                body = SERVICE.work.runtime.start_mujoco()
            elif self.path == '/api/mujoco/stop':
                body = SERVICE.work.runtime.stop_mujoco()
            elif self.path == '/api/gripper':
                body = SERVICE.work.runtime.set_gripper_command(payload.get('command'))
            elif self.path == '/api/teleop/mode':
                body = SERVICE.work.runtime.set_teleop_mode(payload.get('mode'))
            elif self.path == '/api/shutdown':
                body = {'ok': True, 'message': 'Console shutdown requested.'}
                self._json(200, body)
                threading.Thread(target=_shutdown_console, daemon=True).start()
                return
            elif self.path == '/api/ui-session/heartbeat':
                UI_SESSIONS.heartbeat(payload.get('client_id'))
                body = {'ok': True}
            elif self.path == '/api/ui-session/close':
                UI_SESSIONS.close(payload.get('client_id'))
                body = {'ok': True}
            elif self.path == '/api/work/start':
                body = SERVICE.work.start(WorkConfig(**payload))
            elif self.path == '/api/work/output-folder':
                handler = getattr(SERVICE.work.runtime, 'set_output_folder', None)
                if handler is None:
                    raise ValueError('Folder selection is not enabled for this collection profile')
                body = handler(payload.get('path'))
            elif self.path == '/api/work/action':
                body = SERVICE.work.action(str(payload.get('action', '')))
            elif self.path == '/api/work/weight':
                body = SERVICE.work.runtime.set_weight(payload.get('weight'), payload.get('recording_id'), payload.get('request_id'))
            elif self.path == '/api/train/start':
                body = SERVICE.train.start(**payload)
            elif self.path == '/api/experiment/folders/create':
                body = create_folder(payload.get('parent'), str(payload.get('name', '')))
            elif self.path == '/api/folder-picker':
                body = SERVICE.folder_picker.select(str(payload.get('purpose', '')), payload.get('initial_path'))
            elif self.path == '/api/train/stop':
                body = SERVICE.train.stop()
            elif self.path == '/api/conversion/start':
                body = SERVICE.conversion.start(**payload)
            elif self.path == '/api/eval/start':
                body = SERVICE.eval.start(**payload)
            elif self.path == '/api/eval/action':
                body = SERVICE.eval.action(str(payload.get('action', '')))
            else:
                self._json(404, {'error': 'not found'})
                return
            self._json(200, body)
        except (TypeError, ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
            self._json(400, {'error': str(error)})

    def _payload(self) -> dict:
        length = int(self.headers.get('Content-Length', '0'))
        if length > 64 * 1024:
            raise ValueError('request body is too large')
        raw = self.rfile.read(length) if length else b'{}'
        value = json.loads(raw.decode('utf-8'))
        if not isinstance(value, dict):
            raise ValueError('request body must be a JSON object')
        return value

    def _json(self, status: int, body: dict) -> None:
        encoded = _encode_json(body)
        self.send_response(status)
        self._headers()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _binary(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._headers()
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _headers(self) -> None:
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    def log_message(self, format: str, *args: object) -> None:
        return

def _shutdown_console() -> None:
    """Stop child runtimes, then let run_ui.sh tear down the frontend too."""
    if SHUTDOWN_REQUESTED.is_set():
        return
    SHUTDOWN_REQUESTED.set()
    if SERVER is not None:
        SERVER.shutdown()

class ConsoleHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    _enable_runtime_profile()
    global SERVER
    server = ConsoleHTTPServer((args.host, args.port), Handler)
    SERVER = server

    def request_shutdown(_signum: int, _frame: object) -> None:
        threading.Thread(target=_shutdown_console, daemon=True).start()
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    def watch_windows():
        while not SHUTDOWN_REQUESTED.wait(0.25):
            if UI_SESSIONS.should_shutdown():
                _shutdown_console()
                return
    threading.Thread(target=watch_windows, daemon=True).start()
    print(f'Experiment console API listening on http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        SERVICE.close()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
