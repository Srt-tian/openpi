#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="pi05_handumi_tblock_piper_joint_lora"
EXPECTED_GPU_COUNT=4
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-handumi_tblock_piper_joint_fsdp4_lora_${TIMESTAMP}}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
OPENPI_EXPECTED_COMMIT="${OPENPI_EXPECTED_COMMIT:?OPENPI_EXPECTED_COMMIT is required}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENDOR_DIR="/workspace/user/dependencies/openpi_handumi_piper"
RUNTIME_PYTHONPATH="$(mktemp -d /tmp/openpi_handumi_python.XXXXXX)"
DATA_PARENT="/workspace/user/datasets"
DATA_ROOT="${DATA_PARENT}/handumi_tblock_all_piper_clean_piper_joints"
BASE_PARAMS="/pfs/user/Models/openpi/pi05_base/params"
NORM_STATS="/pfs/user/Models/openpi_handumi_piper/assets/${CONFIG_NAME}/handumi_tblock_all_piper_clean_piper_joints/norm_stats.json"
LOG_DIR="/pfs/user/experiments/openpi_handumi_piper/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

for required_path in "${DATA_ROOT}" "${BASE_PARAMS}" "${NORM_STATS}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required training input is missing: ${required_path}" >&2
        exit 1
    fi
done

(cd "${VENDOR_DIR}" && sha256sum --check SHA256SUMS)
# The production image intentionally has no pip in /.venv. These are pure-Python
# wheels, so extract them directly into an isolated task-local import directory.
for wheel in \
    "${VENDOR_DIR}/accelerate-1.10.1-py3-none-any.whl" \
    "${VENDOR_DIR}/lerobot-0.4.4-py3-none-any.whl"; do
    /.venv/bin/python -m zipfile -e "${wheel}" "${RUNTIME_PYTHONPATH}"
done

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
    echo "Expected ${EXPECTED_GPU_COUNT} visible GPUs, found ${GPU_COUNT}." >&2
    exit 1
fi

mkdir -p \
    /pfs/user/Models/openpi_handumi_piper/checkpoints \
    /pfs/user/cache/openpi_handumi_piper/root_cache/jax \
    /pfs/user/experiments/openpi_handumi_piper/wandb \
    "${LOG_DIR}"

cd "${REPO_ROOT}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${OPENPI_EXPECTED_COMMIT}" ]]; then
    echo "Expected commit ${OPENPI_EXPECTED_COMMIT}, found ${ACTUAL_COMMIT}." >&2
    exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Training checkout is not clean." >&2
    git status --short >&2
    exit 3
fi
export PYTHONPATH="${RUNTIME_PYTHONPATH}:${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME="${DATA_PARENT}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export WANDB_DIR="${WANDB_DIR:-/pfs/user/experiments/openpi_handumi_piper/wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is required when WANDB_MODE=online." >&2
    exit 2
fi

/.venv/bin/python -c "from openpi.training import config; c=config.get_config('${CONFIG_NAME}'); assert c.fsdp_devices == 4; assert c.batch_size == 8; assert c.ema_decay is None; print(c.name, 'fsdp_devices=', c.fsdp_devices, 'global_batch=', c.batch_size, 'ema=', c.ema_decay)"

echo "run_name=${RUN_NAME}"
echo "config=${CONFIG_NAME}"
echo "dataset=${DATA_ROOT}"
echo "gpu_count=${GPU_COUNT}"
echo "commit=${ACTUAL_COMMIT}"
echo "num_train_steps=${NUM_TRAIN_STEPS}"
echo "log_file=${LOG_FILE}"

set -o pipefail
/.venv/bin/python scripts/train.py "${CONFIG_NAME}" \
    --exp_name="${RUN_NAME}" \
    --project_name="${WANDB_PROJECT:-openpi-handumi-piper}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    --keep-period="${KEEP_PERIOD}" \
    --overwrite \
    2>&1 | tee -a "${LOG_FILE}"
