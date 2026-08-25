#!/usr/bin/env bash
set -euo pipefail

# End-to-end single-GPU Reward Model training pipeline.
#
# Required:
#   export PREFERENCE_DATA_ARCHIVE=/path/to/preference_data.zip
#
# Optional:
#   export MODEL_PATH=models/base-reward-model
#   export TRAINING_CONFIG=configs/training/single_gpu_qlora.json
#   export RESUME_FROM_CHECKPOINT=/path/to/checkpoint

: "${PREFERENCE_DATA_ARCHIVE:?Set PREFERENCE_DATA_ARCHIVE to the preference-data archive.}"

MODEL_PATH="${MODEL_PATH:-models/base-reward-model}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training/single_gpu_qlora.json}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${AUTO_SETUP:-0}" == "1" ]]; then
  "${PROJECT_ROOT}/scripts/setup_gpu_environment.sh"
fi

source .venv/bin/activate

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "===== PREPARE DATA ====="

python -m financial_reward_rl.cli prepare-data \
  --archive "$PREFERENCE_DATA_ARCHIVE" \
  --output-dir data/preferences

python -m financial_reward_rl.cli summarize-data \
  --data data/preferences/train.jsonl


echo
echo "===== PREFLIGHT ====="

python -m financial_reward_rl.cli preflight \
  --train-file data/preferences/train.jsonl \
  --eval-file data/preferences/eval.jsonl \
  --model-path "$MODEL_PATH" \
  --require-ready


echo
echo "===== SINGLE-GPU TRAINING ====="

train_command=(
  python
  -m
  financial_reward_rl.train
  --config "$TRAINING_CONFIG"
  --model-path "$MODEL_PATH"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  train_command+=(
    --resume-from-checkpoint
    "$RESUME_FROM_CHECKPOINT"
  )
fi

"${train_command[@]}"


echo
echo "===== HELD-OUT EVALUATION ====="

OUTPUT_MODEL_PATH="$(
python - <<PY
import json
with open("$TRAINING_CONFIG", encoding="utf-8") as f:
    print(json.load(f)["output_dir"])
PY
)"

MAX_LENGTH="$(
python - <<PY
import json
with open("$TRAINING_CONFIG", encoding="utf-8") as f:
    print(json.load(f)["max_length"])
PY
)"

python -m financial_reward_rl.evaluate \
  --model-path "$OUTPUT_MODEL_PATH" \
  --base-model-path "$MODEL_PATH" \
  --data data/preferences/test.jsonl \
  --max-length "$MAX_LENGTH"

echo
echo "Pipeline complete."
