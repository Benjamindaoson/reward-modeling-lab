#!/usr/bin/env bash
set -euo pipefail

# Reward Modeling Lab V2: shortcut-robust single-GPU experiment.
#
# Required:
#   export PREFERENCE_DATA_ARCHIVE=/path/to/preference_data.zip
#
# Optional:
#   export MODEL_PATH=models/base-reward-model
#   export TRAINING_CONFIG=configs/training/v2_shortcut_robust_1024.json
#   export V2_REPORT_DIR=reports/v2-shortcut-robust
#   export RESUME_FROM_CHECKPOINT=/path/to/checkpoint
#
# This script does not fabricate result artifacts. The promotion gate is run
# only after the V2 model and all three fixed evaluation views complete.

: "${PREFERENCE_DATA_ARCHIVE:?Set PREFERENCE_DATA_ARCHIVE to the preference-data archive.}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base-reward-model}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training/v2_shortcut_robust_1024.json}"
V2_REPORT_DIR="${V2_REPORT_DIR:-reports/v2-shortcut-robust}"

if [[ "${AUTO_SETUP:-0}" == "1" ]]; then
  "${PROJECT_ROOT}/scripts/setup_gpu_environment.sh"
fi

source .venv/bin/activate
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "===== PREPARE ORIGINAL SPLITS ====="
python -m financial_reward_rl.cli prepare-data   --archive "$PREFERENCE_DATA_ARCHIVE"   --output-dir data/preferences

echo
echo "===== BUILD V2 SHORTCUT-ROBUST VIEWS ====="
python -m financial_reward_rl.cli prepare-v2   --train data/preferences/train.jsonl   --eval data/preferences/eval.jsonl   --test data/preferences/test.jsonl   --output-dir data/preferences/v2   --tokenizer-path "$MODEL_PATH"   --matched-relative-gap 0.10   --max-repeat 4   --monitor-size 400   --seed 42

echo
echo "===== PREFLIGHT ====="
python -m financial_reward_rl.cli preflight   --train-file data/preferences/v2/train_balanced_hard.jsonl   --eval-file data/preferences/v2/eval_monitor_400.jsonl   --model-path "$MODEL_PATH"   --require-ready

echo
echo "===== V2 TRAINING ====="
train_command=(
  python
  -m
  financial_reward_rl.train
  --config "$TRAINING_CONFIG"
  --model-path "$MODEL_PATH"
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  train_command+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi
"${train_command[@]}"

OUTPUT_MODEL_PATH="$(
python - <<PY
import json
with open("$TRAINING_CONFIG", encoding="utf-8") as f:
    print(json.load(f)["output_dir"])
PY
)"

echo
echo "===== V2 IID / CHALLENGE EVALUATION ====="
for split in iid length_matched reversed_length; do
  run_dir="$V2_REPORT_DIR/$split"
  mkdir -p "$run_dir"
  python -m financial_reward_rl.evaluation_suite     --model-path "$OUTPUT_MODEL_PATH"     --base-model-path "$MODEL_PATH"     --data "data/preferences/v2/challenges/$split.jsonl"     --run-dir "$run_dir"     --phase finetuned     --max-length 1024     --batch-size 2
done

echo
echo "===== V2 PROMOTION GATE ====="
python -m financial_reward_rl.v2_gate   --v2-iid "$V2_REPORT_DIR/iid/data/finetuned_evaluation.json"   --v2-matched "$V2_REPORT_DIR/length_matched/data/finetuned_evaluation.json"   --v2-reversed "$V2_REPORT_DIR/reversed_length/data/finetuned_evaluation.json"   --output "$V2_REPORT_DIR/promotion_gate.json"   --require-pass

echo
echo "V2 experiment complete and promotion gate passed."
