#!/usr/bin/env bash
set -Eeo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
frontend="$project_root/ui/experiment_console/frontend"
export MIMIC_PYTHON="${MIMIC_PYTHON:-$(command -v python3)}"
export UI_FRONTEND_PORT="${UI_FRONTEND_PORT:-5173}"
export MIMIC_NODE_BIN="${MIMIC_NODE_BIN:-$(dirname -- "$(command -v node)")}"
if [[ -x "$MIMIC_NODE_BIN/node" ]]; then
  export PATH="$MIMIC_NODE_BIN:$PATH"
fi

if [[ "${1:-}" == "--dev" ]]; then
  shift
  if [[ ! -d "$frontend/node_modules" ]]; then
    echo "Install UI dependencies first: cd $frontend && npm ci" >&2
    exit 1
  fi
  cd "$frontend"
  exec npm run dev -- --host 127.0.0.1 "$@"
fi

if [[ ! -d "$frontend/node_modules" ]]; then
  echo "Install UI dependencies first: cd $frontend && npm ci" >&2
  exit 1
fi

# Build the active task profile from source into an ephemeral location. This
# avoids serving a stale checked-in dist and does not modify generated assets.
runtime_dist="/tmp/mimic-experiment-console-${MIMIC_UI_TASK:-pick_and_place}-${UI_FRONTEND_PORT}"
cd "$frontend"
npm run build -- --outDir "$runtime_dist" --emptyOutDir
exec "$MIMIC_PYTHON" -m http.server "$UI_FRONTEND_PORT" \
  --bind 127.0.0.1 --directory "$runtime_dist"
