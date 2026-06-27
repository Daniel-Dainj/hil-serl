#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

cd "${SCRIPT_DIR}"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    "${REPO_ROOT}/.venv/bin/python" ../../train_rlpd.py "$@" \
    --exp_name=workpiece_pickup \
    --checkpoint_path=./rlpd_run \
    --demo_path=./demo_data/*.pkl \
    --learner
