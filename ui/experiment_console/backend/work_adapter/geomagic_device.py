"""Lifecycle owner for the physical Geomagic Touch force-off state streamer.

Used only by the physical Geomagic profile (ui_geo_sq). It owns at most one
child process (geomagic/device/run_hdapi_state_stream.sh), started by a
long-lived supervisor thread and signalled only through its own Popen handle.
No process is ever found or killed by name; another HDAPI owner is reported
and left alone. Health comes from the streamer's own stdout (init lines and
1 Hz status lines); packet freshness itself is verified by the WORK receiver.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, Callable

TOUCH_USB_ID = ("2988", "0302")
STREAM_PORT = 5558
# argv[0] basenames that open the Touch through HDAPI (or the ROS driver).
HDAPI_OWNERS = {"hdapi_state_stream", "hdapi_haptic_servo", "hdapi_force_diagnostic",
                "Touch_Setup", "Touch_Diagnostic", "omni_state", "teleop_geo_hdapi",
                "inspect_openhaptics"}
EXIT_REASONS = {
    2: "hdInitDevice failed: Touch disconnected or opened by another process",
    3: "HDAPI scheduler failed to start",
    4: "calibration not completed: place the stylus fully in the inkwell",
    5: "HDAPI callback could not be scheduled",
    6: "UDP socket could not be created",
    9: "device fault while streaming",
    10: "force output found enabled; streamer refused to run",
}
_STATUS = re.compile(r"(\w+)=(-?[\d.]+)")


def default_command(port: int = STREAM_PORT) -> list[str]:
    override = os.environ.get("MIMIC_GEOMAGIC_STREAMER")  # tests/diagnostics only
    script = Path(override) if override else (
        Path(__import__("geomagic").__file__).resolve().parent / "device" / "run_hdapi_state_stream.sh")
    return [str(script), "--port", str(port), "--status-interval", "1"]


def touch_usb_present(sys_root: Path = Path("/sys/bus/usb/devices")) -> bool:
    try:
        for device in sys_root.iterdir():
            try:
                ids = ((device / "idVendor").read_text().strip(),
                       (device / "idProduct").read_text().strip())
            except OSError:
                continue
            if ids == TOUCH_USB_ID:
                return True
    except OSError:
        return False
    return False


def other_hdapi_owners(exclude: set[int], proc_root: Path = Path("/proc")) -> list[tuple[int, str]]:
    """Read-only scan for processes that may hold the Touch."""
    from geomagic.standalone.device import other_owners
    return other_owners(proc_root, exclude)


def udp_port_free(port: int = STREAM_PORT) -> bool:
    """True when the WORK receiver could bind 127.0.0.1:port right now."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


class GeomagicDeviceSupervisor:
    def __init__(self, command: list[str] | None = None, *, log_path: Path | None = None,
                 restart_delay_s: float = 3.0, stale_after_s: float = 3.0,
                 usb_present: Callable[[], bool] | None = None,
                 other_owners: Callable[[set[int]], list] | None = None,
                 owner_lock_factory: Callable[[], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.command = list(command or default_command())
        self.log_path = Path(log_path or os.environ.get(
            "MIMIC_GEOMAGIC_STREAM_LOG", "/tmp/mimic-geomagic-stream.log"))
        self.restart_delay_s = restart_delay_s
        self.stale_after_s = stale_after_s
        # Resolved at call time so the module-level probes stay patchable.
        self._usb_present = usb_present or (lambda: touch_usb_present())
        self._other_owners = other_owners or (lambda exclude: other_hdapi_owners(exclude))
        self._clock = clock
        self._owner_lock_factory = owner_lock_factory
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._capture_socket: socket.socket | None = None
        self._capture_enabled = bool(os.environ.get("GEOMAGIC_SYNC_CAPTURE"))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._owner_lock = None
        self._reader: threading.Thread | None = None
        self._next_start = 0.0
        self.starts = 0
        self._reset_child_state()
        self._blocked: str | None = None   # reason a start was not attempted
        self._last_exit: int | None = None
        self._force_violation = False
        self.recent_lines: deque[str] = deque(maxlen=50)

    def _reset_child_state(self) -> None:
        self._info: dict[str, str] = {}
        self._streaming = False
        self._calibration_wait = False
        self._last_status_at: float | None = None
        self._previous_status: tuple[float, dict[str, float]] | None = None
        self._rates: dict[str, float] = {}
        self._counters: dict[str, float] = {}
        self._progressing = False

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="geomagic-streamer-supervisor",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()  # SIGTERM to our own child only; streamer shuts down HDAPI
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._reader is not None:
            self._reader.join(timeout=2)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        with self._lock:
            if self._capture_socket is not None:
                self._capture_socket.close()
                self._capture_socket = None
            if self._owner_lock is not None:
                self._owner_lock.__exit__()
                self._owner_lock = None

    def begin_capture(self, directory: Path) -> int:
        """Start a fresh window without reopening the device or resetting sequence."""
        return self._capture_command("BEGIN\t" + str(directory), directory)

    def save_capture(self, directory: Path, pid: int) -> None:
        """Checkpoint paused native buffers while the device owner stays alive."""
        self._capture_command("SAVE", directory, expected_pid=pid)

    def _capture_command(self, command: str, directory: Path, expected_pid: int | None = None) -> int:
        encoded = command.encode()
        if len(encoded) >= 4096 or any(c in str(directory) for c in '\t\r\n'):
            raise ValueError("invalid synchronization capture directory")
        with self._capture_lock:
            with self._lock:
                channel, process = self._capture_socket, self._process
                if channel is None or process is None or process.poll() is not None:
                    raise RuntimeError("native synchronization capture channel is unavailable")
                pid = process.pid
                if expected_pid is not None and pid != expected_pid:
                    raise RuntimeError("native device owner restarted during this capture")
            try:
                channel.send(encoded)
                response = channel.recv(8192).decode()
            except OSError as error:
                # A late acknowledgement must not be consumed by another request.
                with self._lock:
                    if self._capture_socket is channel:
                        self._capture_socket = None
                channel.close()
                raise RuntimeError(f"native synchronization capture failed: {error}") from error
            if response != "OK\t" + str(directory):
                raise RuntimeError(f"native synchronization capture failed: {response}")
            return pid

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def _run(self) -> None:
        # The child is created from this thread so that, with setpriv's parent
        # death signal, a crashed backend also stops the streamer.
        while not self._stop.is_set():
            with self._lock:
                process = self._process
            if process is not None and process.poll() is not None:
                with self._lock:
                    self._last_exit = process.returncode
                    self._process = None
                    self._streaming = False
                    self._next_start = self._clock() + self.restart_delay_s
                    if self._owner_lock is not None:
                        self._owner_lock.__exit__()
                        self._owner_lock = None
            if self._process is None and not self._force_violation and self._clock() >= self._next_start:
                self._try_spawn()
            self._stop.wait(0.25)

    def _try_spawn(self) -> None:
        if not self._usb_present():
            self._blocked = "Touch USB 2988:0302 not detected"
            self._next_start = self._clock() + self.restart_delay_s
            return
        owners = self._other_owners({os.getpid()})
        if owners:
            listed = ", ".join(f"{name} (pid {pid})" for pid, name in owners)
            self._blocked = f"another HDAPI process owns the Touch: {listed}; stop it first"
            self._next_start = self._clock() + self.restart_delay_s
            return
        self._blocked = None
        # Same lock as teleop_geo and the unified UI owner closes launcher races.
        from geomagic.standalone.device import OwnerLock
        owner_lock = (self._owner_lock_factory or OwnerLock)()
        try:
            owner_lock.__enter__()
        except RuntimeError as error:
            self._blocked = str(error)
            self._next_start = self._clock() + self.restart_delay_s
            return
        self._owner_lock = owner_lock
        command = list(self.command)
        parent_channel = child_channel = None
        if self._capture_enabled:
            parent_channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
            parent_channel.settimeout(30.0)
            command.extend(["--sync-control-fd", str(child_channel.fileno())])
        if shutil.which("setpriv"):
            command = ["setpriv", "--pdeathsig", "TERM", *command]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._lock:
                self._reset_child_state()
                self._process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                    pass_fds=(() if child_channel is None else (child_channel.fileno(),)),
                )
                if self._capture_socket is not None:
                    self._capture_socket.close()
                self._capture_socket = parent_channel
                self.starts += 1
                process = self._process
        except BaseException:
            owner_lock.__exit__()
            self._owner_lock = None
            if parent_channel is not None:
                parent_channel.close()
            raise
        finally:
            if child_channel is not None:
                child_channel.close()
        self._reader = threading.Thread(target=self._read, args=(process,),
                                        name="geomagic-streamer-reader", daemon=True)
        self._reader.start()

    # ------------------------------------------------------------- parsing
    def _read(self, process: subprocess.Popen[str]) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(f"--- start pid={process.pid} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                for line in process.stdout:  # type: ignore[union-attr]
                    log.write(line)
                    log.flush()
                    self._parse(line.strip())
        finally:
            process.stdout.close()  # type: ignore[union-attr]  # no fd leak across restarts

    def _parse(self, line: str) -> None:
        if not line or line.startswith("*") or "OPENHAPTICS" in line or "3D Systems to obtain" in line:
            return
        with self._lock:
            self.recent_lines.append(line)
            if line.startswith("status "):
                values = {k: float(v) for k, v in _STATUS.findall(line)}
                now = self._clock()
                if self._previous_status is not None:
                    then, previous = self._previous_status
                    span = max(1e-6, now - then)
                    self._rates = {
                        "hdapi_hz": (values.get("valid", 0) - previous.get("valid", 0)) / span,
                        "publish_hz": (values.get("published", 0) - previous.get("published", 0)) / span,
                    }
                    self._progressing = (values.get("valid", 0) > previous.get("valid", 0)
                                         and values.get("published", 0) > previous.get("published", 0))
                else:
                    self._progressing = values.get("valid", 0) > 0 and values.get("published", 0) > 0
                self._previous_status = (now, values)
                self._counters = values
                self._last_status_at = now
                return
            key, sep, value = line.partition("=")
            if not sep:
                return
            if key == "FORCE_OUTPUT_ENABLED" and value != "NO":
                self._force_violation = True
                process = self._process
                if process is not None and process.poll() is None:
                    process.terminate()
            elif key == "streaming" and value == "STARTED":
                self._streaming = True
                self._calibration_wait = False
            elif key == "calibration_action":
                self._calibration_wait = True
            elif key == "calibration_after" and value == "OK":
                self._calibration_wait = False
            elif key in {"hdInitDevice", "model", "vendor", "serial", "session_id",
                         "calibration_before", "calibration_after", "destination",
                         "FORCE_OUTPUT_ENABLED", "source_clock_domain"}:
                self._info[key] = value

    # -------------------------------------------------------------- status
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            alive = self._process is not None and self._process.poll() is None
            fresh_status = (self._last_status_at is not None
                            and now - self._last_status_at <= self.stale_after_s)
            errors = int(self._counters.get("errors", 0))
            if self._force_violation:
                state, detail = "error", "force output reported enabled; streamer stopped"
            elif alive and self._streaming and fresh_status and self._progressing and errors == 0:
                state, detail = "connected", "physical HDAPI streaming to 127.0.0.1:5558"
            elif alive and self._streaming and self._last_status_at is None:
                state, detail = "waiting", "streaming started; waiting for the first health report"
            elif alive and self._streaming:
                state, detail = "stale", ("HDAPI device errors reported" if errors else
                                          "streamer alive but device samples are not advancing")
            elif alive and self._calibration_wait:
                state, detail = "waiting", "place the stylus fully in the inkwell to calibrate"
            elif alive:
                state, detail = "waiting", "initializing Geomagic Touch (HDAPI)"
            elif self._blocked:
                state, detail = "error" if "another" in self._blocked else "disconnected", self._blocked
            elif self._last_exit is not None:
                reason = EXIT_REASONS.get(self._last_exit, f"streamer exited (code {self._last_exit})")
                state, detail = "error", f"{reason}; retrying"
            else:
                state, detail = "waiting", "starting Geomagic Touch streamer"
            return {
                "state": state, "detail": detail, "device_source": "physical_hdapi",
                "streamer_pid": self._process.pid if alive else None,
                "streamer_starts": self.starts, "last_exit_code": self._last_exit,
                "model": self._info.get("model"), "serial": self._info.get("serial"),
                "session_id": self._info.get("session_id") if alive else None,
                "calibration": self._info.get("calibration_after") or self._info.get("calibration_before"),
                "force_output_enabled": (None if "FORCE_OUTPUT_ENABLED" not in self._info
                                         else self._info["FORCE_OUTPUT_ENABLED"] != "NO"
                                         or self._force_violation),
                "hdapi_hz": round(self._rates["hdapi_hz"], 1) if "hdapi_hz" in self._rates else None,
                "publish_hz": round(self._rates["publish_hz"], 1) if "publish_hz" in self._rates else None,
                "device_errors": errors,
                "log": str(self.log_path),
            }

    def ready(self) -> tuple[bool, str]:
        status = self.snapshot()
        if status["state"] != "connected":
            return False, f"Geomagic Touch is {status['state'].upper()}: {status['detail']}"
        if status["force_output_enabled"] is not False:
            return False, "Geomagic streamer has not confirmed FORCE_OUTPUT_ENABLED=NO"
        return True, "ready"


class GeomagicHapticSupervisor:
    """One unified teleop_geo HDAPI owner; no force-off child in this profile.

    The long-lived HTTP parent owns the process. A WORK child inherits just
    the force socket, so a stalled/crashed WORK cannot hold a nonzero command.
    A device fault requires a new UI session rather than silently reconnecting
    the old force channel. Other device owners are never terminated.
    """
    def __init__(self, *, owner_factory=None, preflight_fn=None, lock_factory=None):
        from geomagic.standalone.device import HapticOwner, OwnerLock, preflight
        self._owner_factory = owner_factory or HapticOwner
        self._preflight = preflight_fn or preflight
        self._lock_factory = lock_factory or OwnerLock
        self.owner = self._owner_lock = None
        self._thread = None
        self._stop = threading.Event()
        self._capture_lock = threading.Lock()
        self.error = None
        self.starts = 0

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='geomagic-haptic-owner', daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._owner_lock = self._lock_factory()
            self._owner_lock.__enter__()
            runtime = self._preflight(state_port=STREAM_PORT)
            if self._stop.is_set():
                return
            self.owner = self._owner_factory(runtime, STREAM_PORT, sequence_clock=True,
                                            capture_control=bool(os.environ.get('GEOMAGIC_SYNC_CAPTURE')))
            self.owner.start()
            self.starts += 1
            self.owner.wait_ready(stop_requested=self._stop.is_set)
            while not self._stop.wait(.1):
                if self.owner.process.poll() is not None:
                    raise RuntimeError('HDAPI owner stopped: ' + self.owner.exit_description())
        except BaseException as error:
            if not self._stop.is_set():
                self.error = str(error)
        finally:
            if self.owner is not None:
                self.owner.close()
            if self._owner_lock is not None:
                self._owner_lock.__exit__()

    @property
    def pid(self):
        p = self.owner.process if self.owner is not None else None
        return p.pid if p is not None and p.poll() is None else None

    def zero_force(self):
        if self.owner is not None and not self.owner.closed:
            self.owner.send((0, 0, 0), enabled=False)

    def stop(self):
        self._stop.set()
        self.zero_force()
        if self._thread is not None:
            self._thread.join(timeout=6)

    def worker_transport(self):
        ready, reason = self.ready()
        if not ready:
            raise RuntimeError(reason)
        # multiprocessing's socket reducer duplicates this endpoint into spawn.
        return {'_haptics_enabled': True, '_haptic_socket': self.owner.sender,
                '_haptic_token': self.owner.token}

    def snapshot(self):
        owner = self.owner
        alive = self.pid is not None
        ready = alive and owner.ready.is_set() and not owner.closed
        status = owner.status if owner is not None else {}
        healthy = (ready and time.monotonic() - status.get('received_at', -float('inf')) < 3.
                   and status.get('servo_hz', 0) > 0 and status.get('state_hz', 0) > 0)
        return dict(state='error' if self.error else 'connected' if healthy else 'waiting',
                    detail=self.error or ('unified physical state + haptic HDAPI' if healthy else 'initializing unified HDAPI / calibration'),
                    device_source='physical_hdapi', streamer_pid=self.pid,
                    streamer_starts=self.starts, last_exit_code=owner.process.poll() if owner and owner.process else None,
                    model=owner.info.get('model') if owner else None,
                    serial=owner.info.get('serial') if owner else None,
                    session_id=f'teleop-geo-{owner.token}' if ready else None,
                    calibration='OK' if ready else None, force_output_enabled=True if ready else None,
                    hdapi_hz=status.get('servo_hz'), publish_hz=status.get('state_hz'),
                    device_errors=int(bool(self.error)), rendered_force_n=status.get('rendered_force_n', 0.),
                    watchdog=('OK' if status.get('watchdog') in {'ACTIVE', 'ZERO'} and healthy else 'STALE'),
                    log=None)

    def ready(self):
        s = self.snapshot()
        return (True, 'ready') if s['state'] == 'connected' else (False, s['detail'])

    def begin_capture(self, directory):
        return self._capture_command('BEGIN\t' + str(directory), directory)

    def save_capture(self, directory, pid):
        if self.pid is None and all((Path(directory) / f'native-{kind}-{pid}.{suffix}').exists()
                                    for kind, suffix in (('meta', 'json'), ('servo', 'csv'), ('packets', 'csv'))):
            return  # Owner shutdown already flushed its bounded native window.
        self._capture_command('SAVE', directory, pid)

    def _capture_command(self, command, directory, expected_pid=None):
        if len(command.encode()) >= 4096 or any(c in str(directory) for c in '\t\r\n'):
            raise ValueError('invalid synchronization capture directory')
        with self._capture_lock:
            owner, pid = self.owner, self.pid
            if owner is None or pid is None or owner.capture_channel is None:
                raise RuntimeError('native haptic capture channel is unavailable')
            if expected_pid is not None and pid != expected_pid:
                raise RuntimeError('native owner changed during capture')
            try:
                owner.capture_channel.send(command.encode())
                reply = owner.capture_channel.recv(8192).decode()
            except OSError as error:
                owner.capture_channel.close()
                owner.capture_channel = None
                raise RuntimeError('native haptic capture failed') from error
            if reply != 'OK\t' + str(directory):
                raise RuntimeError('native haptic capture failed: ' + reply)
            return pid
