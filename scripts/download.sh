#!/usr/bin/env bash
set -euo pipefail

declare -A MODELS=(
  [qwen3-14b]="Qwen/Qwen3-14B-AWQ"
  [qwen3-32b]="Qwen/Qwen3-32B-AWQ"
  [qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ"
  [qwen25-coder-32b]="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
)

usage() {
  echo "Usage: $0 <model-key> | --list"
  echo "Available keys:"
  for k in "${!MODELS[@]}"; do echo "  $k -> ${MODELS[$k]}"; done
  exit 1
}

[[ $# -ne 1 ]] && usage

if [[ "$1" == "--list" ]]; then
  usage
fi

KEY="$1"
MODEL_ID="${MODELS[$KEY]:-}"
[[ -z "$MODEL_ID" ]] && { echo "Unknown model key: $KEY"; usage; }

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="${ROOT_DIR}/models/${MODEL_ID}"
mkdir -p "$LOCAL_DIR"

# 缓存全部收敛到工作空间，避免写 C 盘（~/.cache）
export MODELSCOPE_CACHE="${ROOT_DIR}/.cache/modelscope"
export PIP_CACHE_DIR="${ROOT_DIR}/.cache/pip"
export TMPDIR="${ROOT_DIR}/.cache/tmp"
mkdir -p "$MODELSCOPE_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

if ! command -v modelscope &>/dev/null; then
  echo "modelscope CLI not found, installing..."
  pip3 install --user modelscope
fi

echo "Downloading ${MODEL_ID} -> ${LOCAL_DIR}"
modelscope download --model "$MODEL_ID" --local_dir "$LOCAL_DIR"
echo "Done."
