#!/usr/bin/env bash
set -euo pipefail
: "${OPENPI_DATA_HOME:?must point at existing tokenizer assets}"
: "${WANDB_API_KEY:?must be injected by the task environment}"
repo_dir=${REPO_DIR:?set REPO_DIR to the checked-in V2 train checkout}
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact approved commit}"
actual_commit=$(git -C "$repo_dir" rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || { echo "fatal: HEAD != EXPECTED_COMMIT" >&2; exit 2; }
[[ -z $(git -C "$repo_dir" status --porcelain --untracked-files=all) ]] || { echo "fatal: checkout is dirty" >&2; exit 2; }
if [[ ${PI05_ALLOW_UNPUBLISHED_COMMIT:-0} != 1 ]]; then
  upstream=$(git -C "$repo_dir" rev-parse '@{upstream}')
  [[ $(git -C "$repo_dir" rev-parse "$upstream") == "$EXPECTED_COMMIT" ]] || { echo "fatal: upstream mismatch" >&2; exit 2; }
fi
OPENPI_RUNTIME_OVERLAY=${OPENPI_RUNTIME_OVERLAY:-/pfs/user/data/physicalrsi_pi05/runtime_deps/av15_overlay}
[[ -d $OPENPI_RUNTIME_OVERLAY ]] || { echo "fatal: OPENPI_RUNTIME_OVERLAY is not a directory" >&2; exit 2; }
overlay_prefix="$OPENPI_RUNTIME_OVERLAY:"
export PYTHONPATH="$overlay_prefix$repo_dir:$repo_dir/src:$repo_dir/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin=/.venv/bin/python
[[ -x "$python_bin" ]] || { echo "fatal: /.venv/bin/python missing" >&2; exit 2; }
"$python_bin" -c 'import av, flax, jax, numpy, optax, pyarrow, torch, wandb'
exec "$python_bin" "$repo_dir/scripts/train_four_residual_heads.py" \
  --base-checkpoint "${BASE_CHECKPOINT:?}" \
  --base-params-sha256 "${BASE_PARAMS_SHA256:?}" \
  --data-root "${DATA_ROOT:?}" \
  --output-dir "${OUTPUT_DIR:?}" \
  "$@"
