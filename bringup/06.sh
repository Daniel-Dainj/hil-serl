#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}/examples"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    "${REPO_ROOT}/.venv/bin/python" record_success_fail.py \
    --exp_name workpiece_pickup \
    --successes_needed 500 \
    "$@"
