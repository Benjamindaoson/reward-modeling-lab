# V2 — Shortcut-Robust Reward Model

## Status

**Implementation: READY**  
**8B GPU run: NOT EXECUTED / NOT REPORTED**

V2 is intentionally a narrow intervention. It does not attempt every item from the original roadmap. It tests one question:

> Can we reduce response-length shortcut reliance while preserving useful IID preference ranking?

## Frozen V2 intervention

V2 changes exactly three things relative to the verified V1 experiment:

1. **Length-balanced training exposure**
   - build three strata from existing preference labels:
     - length-matched;
     - preferred response longer;
     - preferred response shorter;
   - deterministically rebalance exposure across the three strata;
   - cap source-example repetition with a configurable max-repeat rule.

2. **Natural anti-length hard negatives**
   - operational definition: preferred/chosen response is shorter than rejected response;
   - no new correctness labels are generated;
   - no LLM-generated preference labels are introduced;
   - these pairs are the auditable implementation of the intended concise-preferred / verbose-rejected intervention.

3. **1024-token context**
   - V1: 512 tokens;
   - V2: 1024 tokens;
   - per-device batch is reduced from 8 to 4;
   - gradient accumulation is increased from 1 to 2;
   - effective batch target remains 8.

Frozen config:

configs/training/v2_shortcut_robust_1024.json

## Data contract

The V2 builder is:

src/financial_reward_rl/v2_data.py

Production preparation should use the reward model tokenizer:

    python -m financial_reward_rl.cli prepare-v2 \
      --train data/preferences/train.jsonl \
      --eval data/preferences/eval.jsonl \
      --test data/preferences/test.jsonl \
      --output-dir data/preferences/v2 \
      --tokenizer-path models/base-reward-model \
      --matched-relative-gap 0.10 \
      --max-repeat 4 \
      --monitor-size 400 \
      --seed 42

Generated local artifacts:

    data/preferences/v2/
    ├── train_balanced_hard.jsonl
    ├── eval_monitor_400.jsonl
    ├── manifest.json
    └── challenges/
        ├── iid.jsonl
        ├── length_matched.jsonl
        └── reversed_length.jsonl

The manifest records:

- length metric/tokenizer;
- pre/post bucket counts;
- anti-length exposure before/after balancing;
- repetition count;
- fixed challenge-set sizes;
- 512-vs-1024 truncation audit.

Raw/private preference data and generated V2 datasets remain excluded from Git.

## Evaluation contract

V2 is evaluated on the same three conceptual views used to interpret V1:

- full frozen IID test;
- length-matched challenge;
- reversed-length challenge.

The end-to-end runner is:

scripts/run_v2_on_gpu.sh

It performs:

    prepare original splits
        ↓
    build V2 balanced/hard-negative view
        ↓
    preflight
        ↓
    8B QLoRA at 1024 tokens
        ↓
    IID evaluation
        ↓
    length-matched evaluation
        ↓
    reversed-length evaluation
        ↓
    V2 promotion gate

## Promotion rule

V2 is **not** promoted because IID accuracy is higher.

The default gate in src/financial_reward_rl/v2_gate.py requires:

- IID regression no worse than **-5 percentage points**;
- Length-Matched improvement at least **+5 percentage points**;
- Reversed-Length improvement at least **+5 percentage points**.

This deliberately permits a result such as:

    IID:              91% -> 89%
    Reversed-Length:  75% -> 85%+

because the target is reward validity under intervention, not IID optimization.

The 85% figure above is an illustrative target, **not an observed result**.

## Claim boundary

Until a real V2 GPU run is completed and retained:

- do not report V2 accuracy;
- do not claim the shortcut is fixed;
- do not claim 1024 tokens solved truncation;
- do not claim robustness improved.

What is currently implemented and testable is the **experimental protocol**, dataset-construction logic, 1024-token training contract, fixed challenge evaluation path, and promotion gate.
