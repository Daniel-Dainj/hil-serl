#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"

ROBOT_IP="${ROBOT_IP:-192.168.1.110}"
DESK_HOST="${DESK_HOST:-${ROBOT_IP}}"
DESK_USER="${DESK_USER:-franka}"
DESK_PASS="${DESK_PASS:-franka123}"
GRIPPER_TYPE="${GRIPPER_TYPE:-Franka}"
FLASK_URL="${FLASK_URL:-0.0.0.0}"
ROS_PORT="${ROS_PORT:-11511}"
AUTO_ACTIVATE_FCI="${AUTO_ACTIVATE_FCI:-1}"
FORCE_TAKE_CONTROL="${FORCE_TAKE_CONTROL:-1}"
AUTO_SETUP_PYTHON="${AUTO_SETUP_PYTHON:-}"

source "${WORKSPACE_DIR}/catkin_ws/devel/setup.bash"

find_auto_setup_python() {
  local candidate
  local candidates=()

  if [[ -n "${AUTO_SETUP_PYTHON}" ]]; then
    candidates+=("${AUTO_SETUP_PYTHON}")
  else
    candidates+=(
      "${HOME}/panda-py-venv/bin/python"
      "${REPO_DIR}/.venv/bin/python"
      "$(command -v python3)"
    )
  fi

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if "${candidate}" -c 'import panda_py' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

if [[ "${AUTO_ACTIVATE_FCI}" != "0" ]]; then
  if ! AUTO_SETUP_PYTHON="$(find_auto_setup_python)"; then
    echo "Desk auto-setup requires panda_py, but no usable Python interpreter was found." >&2
    echo "Set AUTO_SETUP_PYTHON=/path/to/python where 'import panda_py' succeeds," >&2
    echo "or skip Desk auto-setup with AUTO_ACTIVATE_FCI=0." >&2
    exit 1
  fi

  DESK_ARGS=(
    --desk-host "${DESK_HOST}"
    --desk-user "${DESK_USER}"
    --desk-pass "${DESK_PASS}"
  )
  if [[ "${FORCE_TAKE_CONTROL}" == "0" ]]; then
    DESK_ARGS+=(--no-force-take)
  fi

  "${AUTO_SETUP_PYTHON}" "${REPO_DIR}/bringup/prepare_panda_desk.py" "${DESK_ARGS[@]}"
fi

export ROS_MASTER_URI="http://localhost:${ROS_PORT}"

cd "${REPO_DIR}"
echo "Starting Franka server on ${FLASK_URL}:5000 (robot clients may still use 127.0.0.2 if desired)"
uv run python serl_robot_infra/robot_servers/franka_server.py \
  --robot_ip="${ROBOT_IP}" \
  --gripper_type="${GRIPPER_TYPE}" \
  --flask_url="${FLASK_URL}" \
  --ros_port="${ROS_PORT}"
