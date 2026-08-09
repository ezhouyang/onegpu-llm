#!/usr/bin/env bash
set -euo pipefail

UI_DIR="$(dirname "$0")/../ui"
VENV="${UI_DIR}/.venv"
ROOT_DIR="$(cd "${UI_DIR}/.." && pwd)"

export PIP_CACHE_DIR="${ROOT_DIR}/.cache/pip"

if [[ ! -d "$VENV" ]]; then
  echo "首次运行，创建虚拟环境..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "${UI_DIR}/requirements.txt"
fi

echo "管理界面: http://localhost:9000"
exec "$VENV/bin/python" "${UI_DIR}/server.py"
