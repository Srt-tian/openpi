#!/usr/bin/env bash
set -euo pipefail
: "${OPENPI_DATA_HOME:?must point at existing tokenizer assets}"
: "${WANDB_API_KEY:?must be injected by the task environment}"
repo_dir=${REPO_DIR:?set REPO_DIR to the checked-in V2 train checkout}
export PYTHONPATH="$repo_dir/src:$repo_dir/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$repo_dir/scripts/train_four_residual_heads.py" \
  --base-checkpoint "${BASE_CHECKPOINT:?}" \
  --data-root "${DATA_ROOT:?}" \
  --output-dir "${OUTPUT_DIR:?}" \
  "$@"
