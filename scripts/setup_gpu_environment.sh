#!/usr/bin/env bash
set -euo pipefail

# Required on the GPU machine, for example:
# export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
: "${TORCH_INDEX_URL:?Set TORCH_INDEX_URL to the CUDA-compatible PyTorch wheel index.}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable: install a compatible NVIDIA driver before continuing." >&2
  exit 1
fi
nvidia-smi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX_URL"
python -m pip install -e '.[train]'
