#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

cd "${SCRIPT_DIR}"

# Keep the robot-facing actor on CPU by default so the learner can own the GPU.
: "${JAX_PLATFORMS:=cpu}"
if [[ "${JAX_PLATFORMS}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.05 \
JAX_PLATFORMS="${JAX_PLATFORMS}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    "${REPO_ROOT}/.venv/bin/python" ../../train_rlpd.py "$@" \
    --exp_name=workpiece_pickup \
    --checkpoint_path=./rlpd_run \
    --actor
