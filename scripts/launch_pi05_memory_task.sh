#!/usr/bin/env bash
set -euo pipefail
repo_dir=${REPO_DIR:?set the dedicated memory training checkout}
python_bin=${PI05_PYTHON:-/.venv/bin/python}
[[ -x "$python_bin" ]] || { echo "missing memory Python runtime" >&2; exit 2; }
export PYTHONPATH="${OPENPI_RUNTIME_OVERLAY:+$OPENPI_RUNTIME_OVERLAY:}$repo_dir:$repo_dir/src:$repo_dir/scripts:$repo_dir/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$repo_dir/scripts/run_pi05_memory_task.py" \
  --config "${MEMORY_CONFIG:-$repo_dir/configs/pi05_memory_task.json}" "$@"
