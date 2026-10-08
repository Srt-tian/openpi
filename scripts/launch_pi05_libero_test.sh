#!/usr/bin/env bash
set -euo pipefail

classify_launch_args() {
  PI05_FM_ONLY=false
  PI05_HAS_ROLLOUT_MANIFEST=false
  PI05_HAS_REQUIRE_JOINT=false
  for arg in "$@"; do
    case "${arg}" in
      --fm-only) PI05_FM_ONLY=true ;;
      --rollout-manifest|--rollout-manifest=*) PI05_HAS_ROLLOUT_MANIFEST=true ;;
      --require-joint-losses) PI05_HAS_REQUIRE_JOINT=true ;;
    esac
  done
}

validate_launch_mode() {
  if [[ "${PI05_FM_ONLY}" == true ]]; then
    if [[ "${PI05_HAS_ROLLOUT_MANIFEST}" == true ]]; then
      echo "fatal: --fm-only cannot be combined with --rollout-manifest" >&2
      return 2
    fi
    if [[ "${PI05_HAS_REQUIRE_JOINT}" == true ]]; then
      echo "fatal: --fm-only cannot be combined with --require-joint-losses" >&2
      return 2
    fi
    PI05_LAUNCH_MODE=explicit_fm_only
    return 0
  fi
  if [[ "${PI05_HAS_ROLLOUT_MANIFEST}" != true ]]; then
    echo "fatal: named joint-loss job requires --rollout-manifest with full four-target handoff and uncensored call coverage; use --fm-only only for an explicitly approved Stage A run" >&2
    return 2
  fi
  PI05_LAUNCH_MODE=required_joint_losses
}

build_runner_argv() {
  PI05_RUNNER_ARGS=("$@")
  if [[ "${PI05_LAUNCH_MODE}" == required_joint_losses && "${PI05_HAS_REQUIRE_JOINT}" != true ]]; then
    PI05_RUNNER_ARGS+=(--require-joint-losses)
  fi
}

# Sourcing exposes only the argument helpers used by dependency-free regression
# tests.  Executing the launcher still runs every Git and environment gate below.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

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
classify_launch_args "$@"
validate_launch_mode
build_runner_argv "$@"
exec "${python_bin}" "${runner}" "${PI05_RUNNER_ARGS[@]}"
