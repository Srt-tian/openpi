#!/usr/bin/env bash
set -euo pipefail

: "${REPO_DIR:?set REPO_DIR to the frozen OpenPI checkout}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact approved commit}"
: "${OPENPI_DATA_HOME:?set OPENPI_DATA_HOME to the offline asset root}"

if [[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "fatal: REPO_DIR HEAD does not match EXPECTED_COMMIT" >&2; exit 2
fi
git -C "${REPO_DIR}" diff --quiet --ignore-submodules --
git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules --
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=normal)" ]]; then
  echo "fatal: REPO_DIR is not clean" >&2; exit 2
fi
upstream="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"
[[ -n "${upstream}" ]] || { echo "fatal: branch has no upstream" >&2; exit 2; }
if [[ "$(git -C "${REPO_DIR}" rev-parse "${upstream}")" != "${EXPECTED_COMMIT}" ]]; then
  echo "fatal: upstream commit does not match EXPECTED_COMMIT" >&2; exit 2
fi

python_bin=/.venv/bin/python
[[ -x "${python_bin}" ]] || { echo "fatal: ${python_bin} missing" >&2; exit 2; }

overlay_prefix=""
if [[ -n "${OPENPI_RUNTIME_OVERLAY:-}" ]]; then
  [[ -d "${OPENPI_RUNTIME_OVERLAY}" ]] || { echo "fatal: OPENPI_RUNTIME_OVERLAY is not a directory" >&2; exit 2; }
  overlay_prefix="${OPENPI_RUNTIME_OVERLAY}:"
fi
export PYTHONPATH="${overlay_prefix}${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
"${python_bin}" -c 'import av, flax, jax, numpy, optax, pyarrow, torch, wandb'

runner="${REPO_DIR}/scripts/train_four_plugins.py"
[[ -f "${runner}" ]] || runner="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_four_plugins.py"
has_rollout_manifest=false
for arg in "$@"; do
  if [[ "${arg}" == "--rollout-manifest" || "${arg}" == --rollout-manifest=* ]]; then
    has_rollout_manifest=true
    break
  fi
done
if [[ "${has_rollout_manifest}" != true ]]; then
  echo "fatal: named joint-loss job requires --rollout-manifest with full four-target handoff and uncensored call coverage" >&2
  exit 2
fi
exec "${python_bin}" "${runner}" --require-joint-losses "$@"
