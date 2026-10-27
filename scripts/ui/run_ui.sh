#!/usr/bin/env bash
set -Eeuo pipefail

# The current user collection is the default for the familiar Square command.
# --canonical keeps an explicit entry point to the preserved old XY50 profile.
if [[ "$(basename -- "$0")" == "ui_geo_sq" && "${1:-}" == "--canonical" ]]; then
  shift
elif [[ "$(basename -- "$0")" == "ui_geo_sq" && ( $# == 0 || "${1:-}" == "--print-defaults" || "${1:-}" == "--weight-data" ) ]]; then
  [[ "$(basename -- "$0")" == "ui_geo_sq" ]] || { echo '--weight-data requires ui_geo_sq' >&2; exit 2; }
  if [[ "${1:-}" == "--weight-data" ]]; then shift; fi
  weight_script_dir="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
  weight_workspace="$(cd -- "$weight_script_dir/../.." && pwd)"
  export MIMIC_WEIGHT_COLLECTION=1
  export MIMIC_COLLECTION_WORKSPACE="$weight_workspace/outputs/research/touch_collection_console"
  export UI_BACKEND_PORT="${UI_BACKEND_PORT:-8000}" UI_FRONTEND_PORT="${UI_FRONTEND_PORT:-5173}"
  export MIMIC_MANUAL_LAUNCH=1
fi

# Fixed environment entry point for the experiment console UI.
launcher_name="$(basename -- "$0")"
if [[ "${MIMIC_MANUAL_LAUNCH:-0}" == "1" ]]; then launcher_name=ui_manual; fi
if [[ "$launcher_name" != "ui_manual" ]]; then
  unset MIMIC_COLLECTION_WORKSPACE VITE_MIMIC_PROCESSED_ROOT VITE_MIMIC_TRAINING_ROOT
  unset MIMIC_WEIGHT_COLLECTION VITE_MIMIC_WEIGHT_COLLECTION
fi
script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
project_root="$(cd -- "$(dirname -- "$script_path")/../.." && pwd)"
ui_leader="omy"
ui_input_backend="zmq"
ui_raw_root=""
ui_geomagic_source=""
case "$launcher_name" in
  ui_manual)
    ui_task="mimicgen_square"
    ui_experiment="$ui_task"
    ui_dataset_version=geo_sq_weight_v1
    ui_leader=geomagic
    ui_input_backend=geomagic
    ui_geomagic_source=physical_hdapi
    ui_raw_root=raw
    export VITE_MIMIC_WEIGHT_COLLECTION=1
    export VITE_MIMIC_PROCESSED_ROOT=processed
    export VITE_MIMIC_TRAINING_ROOT=training
    if [[ "$ui_geomagic_source" == "synthetic_udp" ]]; then
      ui_raw_root=raw/_test
      export VITE_MIMIC_PROCESSED_ROOT=processed/_test
      export VITE_MIMIC_TRAINING_ROOT=training/_test
    fi
    export MIMIC_GEOMAGIC_HAPTICS="${MIMIC_GEOMAGIC_HAPTICS:-1}"
    ;;
  ui_geo)
    # Synthetic/debug Geomagic: localhost UDP from a manually started sender.
    # Its own raw root keeps synthetic episodes out of physical seed accounting.
    ui_task="mimicgen_square"
    ui_experiment="mimicgen_square"
    ui_dataset_version="geomagic_fr3_mimicgen_square_operator_wrist_20hz_v1"
    ui_leader="geomagic"
    ui_input_backend="geomagic"
    ui_geomagic_source="synthetic_udp"
    ui_raw_root="mimicgen_square/datasets/raw/restricted30_geomagic_synthetic"
    ;;
  ui_geo_sq)
    # Physical Touch demonstrations; haptics default ON, explicit 0 selects
    # the force-off state streamer instead of the unified HDAPI owner.
    ui_task="mimicgen_square"
    ui_experiment="mimicgen_square"
    ui_dataset_version="geomagic_fr3_mimicgen_square_operator_wrist_20hz_v1"
    ui_leader="geomagic"
    ui_input_backend="geomagic"
    ui_geomagic_source="physical_hdapi"
    ui_raw_root="mimicgen_square/datasets/raw/restricted30_geomagic"
    if [[ -n "${MIMIC_UI_WORKSPACE_VALIDATION:-}" ]]; then
      if [[ ! "$MIMIC_UI_WORKSPACE_VALIDATION" =~ ^square_geomagic_xy50_v[1-9][0-9]*$ ]]; then
        echo 'ui_geo_sq: invalid workspace validation plan' >&2
        exit 2
      fi
      ui_raw_root="mimicgen_square/analysis/workspace_validation/$MIMIC_UI_WORKSPACE_VALIDATION"
      unset MIMIC_UI_COLLECTION_PLAN
    elif [[ -z "${MIMIC_UI_COLLECTION_PLAN:-}" && -f "$project_root/experiments/mimicgen_square/configs/collection/square_geomagic_xy50_v1.frozen.json" ]]; then
      export MIMIC_UI_COLLECTION_PLAN=square_geomagic_xy50_v1
    fi
    # Until XY50 is frozen, Restricted30 stays the default.
    if [[ -z "${MIMIC_UI_WORKSPACE_VALIDATION:-}" && "${MIMIC_UI_COLLECTION_PLAN:-square_restricted30_v1}" != "square_restricted30_v1" ]]; then
      if [[ ! "${MIMIC_UI_COLLECTION_PLAN}" =~ ^square_geomagic_xy50_v[1-9][0-9]*$ ]]; then
        echo 'ui_geo_sq: collection plan must be Restricted30 or versioned XY50 training' >&2
        exit 2
      fi
      ui_raw_root="$ui_raw_root/$MIMIC_UI_COLLECTION_PLAN"
    fi
    ;;
  ui_sq)
    ui_task="mimicgen_square"
    ui_experiment="mimicgen_square"
    ui_dataset_version="omy_fr3_mimicgen_square_operator_wrist_20hz_v1"
    ;;
  *)
    ui_task="pick_and_place"
    ui_experiment="pick_and_place"
    ui_dataset_version="omy_fr3_operator_wrist_20hz_v1"
    ;;
esac

# Force-OFF WORK sessions keep automatic latency capture; haptic traces are
# opt-in. Each captured START gets a fresh subdirectory/window in the backend.
if [[ "$launcher_name" == "ui_geo_sq" ]]; then
  export MIMIC_GEOMAGIC_HAPTICS="${MIMIC_GEOMAGIC_HAPTICS:-1}"
  # Haptic high-rate traces are opt-in; keep the established OFF baseline.
  geo_sync_default=1
  if [[ "${MIMIC_GEOMAGIC_HAPTICS:-0}" == "1" && -z "${GEOMAGIC_SYNC_CAPTURE:-}" ]]; then
    geo_sync_default=0
  fi
  if [[ "${GEOMAGIC_SYNC_AUTO:-$geo_sync_default}" == "0" ]]; then
    unset GEOMAGIC_SYNC_CAPTURE
  else
    export GEOMAGIC_SYNC_CAPTURE="${GEOMAGIC_SYNC_CAPTURE:-$project_root/outputs/geomagic/captures/sync/ui_geo_sq}"
    export GEOMAGIC_SYNC_SECONDS="${GEOMAGIC_SYNC_SECONDS:-180}"
    export GEOMAGIC_SYNC_RECORDS="${GEOMAGIC_SYNC_RECORDS:-450000}"
  fi
fi

print_defaults=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --omy-calibration) export MIMIC_CALIBRATION_ONLY=1 ;;
    --print-defaults) print_defaults=1 ;;
    *) echo "$launcher_name: unsupported argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Launcher identity is authoritative. In particular, stale caller values of
# MIMIC_TASK cannot switch the selected task or its storage roots.
export MIMIC_UI_LEADER="$ui_leader"
export MIMIC_UI_GEOMAGIC_SOURCE="$ui_geomagic_source"
export VITE_MIMIC_LEADER="$ui_leader"
export VITE_MIMIC_INPUT_BACKEND="$ui_input_backend"
export VITE_MIMIC_DATASET_OUTPUT_ROOT="$ui_raw_root"
export MIMIC_UI_TASK="$ui_task"
export MIMIC_TASK="$ui_task"
export MIMIC_UI_EXPERIMENT="$ui_experiment"
export MIMIC_UI_DATASET_VERSION="$ui_dataset_version"
export VITE_MIMIC_TASK="$ui_task"
export VITE_MIMIC_EXPERIMENT="$ui_experiment"
export VITE_MIMIC_DATASET_VERSION="$ui_dataset_version"

if [[ "$print_defaults" == "1" ]]; then
  python3 "$project_root/scripts/ui/print_ui_defaults.py"
  exit 0
fi
if [[ "$launcher_name" == "ui_manual" ]]; then
  PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import json, os
from runtime.collection_workspace import research_root
root = research_root()
identity = dict(task=os.environ['MIMIC_UI_TASK'],
                geomagic_source=os.environ['MIMIC_UI_GEOMAGIC_SOURCE'],
                schema='touch_collection_v1')
root.mkdir(parents=True, exist_ok=True)
manifest = root / 'session.json'
if manifest.exists():
    if json.loads(manifest.read_text()) != identity:
        raise ValueError('Collection session belongs to a different task or device source')
else:
    with manifest.open('x') as stream:
        json.dump(identity, stream, indent=2)
        stream.write('\n')
for name in ('raw', 'processed', 'training', 'runs'):
    (root / name).mkdir(exist_ok=True)
if identity['geomagic_source'] == 'synthetic_udp':
    for name in ('raw', 'processed', 'training'):
        (root / name / '_test').mkdir(exist_ok=True)
PY
  export UI_BACKEND_PORT="${UI_BACKEND_PORT:-8014}"
  export UI_FRONTEND_PORT="${UI_FRONTEND_PORT:-5187}"
  export UI_BACKEND_LOG="${UI_BACKEND_LOG:-$MIMIC_COLLECTION_WORKSPACE/runs/ui-backend.log}"
  export UI_FRONTEND_LOG="${UI_FRONTEND_LOG:-$MIMIC_COLLECTION_WORKSPACE/runs/ui-frontend.log}"
  export UI_BROWSER_LOG="${UI_BROWSER_LOG:-$MIMIC_COLLECTION_WORKSPACE/runs/ui-browser.log}"
  export HF_HOME="${HF_HOME:-$MIMIC_COLLECTION_WORKSPACE/cache/huggingface}"
fi
export MIMIC_PYTHON="${MIMIC_PYTHON:-$(command -v python3)}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/omy-ros-log}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$project_root/environments/mimic_robot:$project_root:${FRANKA_TELEOP_ROOT:-$project_root/../franka-teleoperation}${PYTHONPATH:+:$PYTHONPATH}"
export UI_BACKEND_PORT="${UI_BACKEND_PORT:-8000}"
export UI_FRONTEND_PORT="${UI_FRONTEND_PORT:-5173}"
export VITE_API_ROOT="http://127.0.0.1:$UI_BACKEND_PORT"

if [[ ! -x "$MIMIC_PYTHON" ]]; then
    echo "$launcher_name: Python not found: $MIMIC_PYTHON" >&2
  exit 1
fi

if [[ "${MIMIC_WEIGHT_COLLECTION:-0}" == "1" ]] && "$MIMIC_PYTHON" - "$UI_BACKEND_PORT" "$UI_FRONTEND_PORT" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/api/state?view=work", timeout=1) as response:
        work=json.load(response).get('work',{})
    if (work.get('task')!='mimicgen_square'
            or work.get('manual_collection') is not True):
        raise ValueError('Different collection UI')
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[2]}/", timeout=1).close()
except (OSError,ValueError):
    sys.exit(1)
PY
then
  frontend_url="http://127.0.0.1:$UI_FRONTEND_PORT"
  echo "ui_geo_sq is already running: $frontend_url (manual folder collection)"
  if [[ "${UI_AUTO_OPEN:-1}" == "1" ]]; then
    "$MIMIC_PYTHON" "$project_root/scripts/ui/open_console_window.py" "$frontend_url" "http://127.0.0.1:$UI_BACKEND_PORT" --reopen >"$UI_BROWSER_LOG" 2>&1 || \
      echo "ui_geo_sq: open $frontend_url manually" >&2
  fi
  exit 0
fi

if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq ":(${UI_BACKEND_PORT}|${UI_FRONTEND_PORT})$"; then
    echo "$launcher_name: port ${UI_BACKEND_PORT} or ${UI_FRONTEND_PORT} is already in use; stop the old UI first" >&2
    exit 1
  fi
fi

backend_log="${UI_BACKEND_LOG:-/tmp/mimic-ui-backend.log}"
frontend_log="${UI_FRONTEND_LOG:-/tmp/mimic-ui-frontend.log}"
browser_log="${UI_BROWSER_LOG:-/tmp/mimic-ui-browser.log}"
frontend_url="http://127.0.0.1:$UI_FRONTEND_PORT"

cleanup() {
  trap - EXIT INT TERM
  [[ -n "${browser_opener_pid:-}" ]] && kill "$browser_opener_pid" 2>/dev/null || true
  [[ -n "${backend_pid:-}" ]] && kill "$backend_pid" 2>/dev/null || true
  [[ -n "${frontend_pid:-}" ]] && kill "$frontend_pid" 2>/dev/null || true
  # Reap every owned process. Sending SIGTERM alone left the HTTP process and
  # renderer descendants orphaned while asynchronous cleanup was still running.
  [[ -n "${browser_opener_pid:-}" ]] && wait "$browser_opener_pid" 2>/dev/null || true
  [[ -n "${backend_pid:-}" ]] && wait "$backend_pid" 2>/dev/null || true
  [[ -n "${frontend_pid:-}" ]] && wait "$frontend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

open_browser_when_ready() {
  for _attempt in {1..100}; do
    if "$MIMIC_PYTHON" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=0.5).close()' "$frontend_url" >/dev/null 2>&1; then
      if command -v xdg-open >/dev/null 2>&1; then
        exec "$MIMIC_PYTHON" "$project_root/scripts/ui/open_console_window.py" "$frontend_url" "http://127.0.0.1:$UI_BACKEND_PORT" >"$browser_log" 2>&1
      else
        echo "ui: xdg-open is unavailable; open $frontend_url manually" >&2
      fi
      return
    fi
    sleep 0.1
  done
  echo "ui: frontend did not become ready at $frontend_url" >&2
}

"$project_root/scripts/runtime/run_ui_backend.sh" --host 127.0.0.1 --port "$UI_BACKEND_PORT" >"$backend_log" 2>&1 &
backend_pid=$!
"$project_root/scripts/runtime/run_ui_frontend.sh" >"$frontend_log" 2>&1 &
frontend_pid=$!

if [[ "${UI_AUTO_OPEN:-1}" == "1" ]]; then
  open_browser_when_ready &
  browser_opener_pid=$!
fi

echo "UI backend:  http://127.0.0.1:$UI_BACKEND_PORT"
echo "UI frontend: $frontend_url"
echo "Backend log:  $backend_log"
echo "Frontend log: $frontend_log"

wait -n "$backend_pid" "$frontend_pid"
status=$?
exit "$status"
