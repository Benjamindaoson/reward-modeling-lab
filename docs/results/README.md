# Experimental Results

This directory contains a curated public subset of the verified V1
Reward Modeling experiment.

Large model artifacts, private/raw training data, optimizer states,
and local experiment archives are intentionally excluded.

## 1. Formal training

The formal run used:

- Skywork Reward Llama 3.1 8B
- NVIDIA A10 23GB
- 4-bit NF4 QLoRA
- BF16 compute
- 1,000 optimizer steps
- effective batch size 8
- approximately 3h18m wall-clock time
- approximately 97.7% mean GPU utilization
- approximately 11.85GB peak allocated training VRAM

The training split contains 35,990 preference pairs. The fixed
1,000-step run processed approximately 8,000 pair instances.

![Training loss](figures/training_loss.png)

![Training pairwise accuracy](figures/training_pairwise_accuracy.png)

---

## 2. Frozen held-out result

The held-out test contains:

- 451 independent questions
- 4,510 pairwise comparisons
- 2,255 unique responses

| Model | Pairwise Accuracy |
|---|---:|
| Base Reward Model | 50.42% |
| Fine-tuned Reward Model | **91.35%** |

Absolute improvement:

**+40.93 percentage points**

![Base vs fine-tuned](figures/base_vs_finetuned_accuracy.png)

---

## 3. Quality-gap analysis

The model performs better when the underlying quality difference
between the preferred and rejected responses is larger.

| Quality Gap | Base | Fine-tuned |
|---:|---:|---:|
| 1 | 48.73% | **85.64%** |
| 2 | 50.85% | **94.60%** |
| 3 | 52.00% | **95.68%** |
| 4 | 52.77% | **95.79%** |

![Quality gap accuracy](figures/quality_gap_accuracy.png)

---

## 4. Monitor dynamics

Validation accuracy reaches an approximate plateau during the second
half of training.

![Monitor eval loss](figures/monitor_eval_loss.png)

![Monitor eval accuracy](figures/monitor_eval_accuracy.png)

The trainer selected step 800 because it achieved the lowest
validation loss.

---

## 5. Ranking evaluation

The 451 test questions were reconstructed as 5-way response-ranking
problems.

Selected-model results:

| Metric | Result |
|---|---:|
| Kendall tau | **0.8828** |
| NDCG@5 | **0.9474** |
| Perfect 5-way ranking | **57.87%** |
| Level-5 response ranked first | **63.86%** |
| Level-1 response ranked last | **95.57%** |

These metrics expose ranking quality that pairwise accuracy alone
cannot show.

---

## 6. Shortcut audit

The original V1 preference data contains a strong response-length
artifact.

A trivial heuristic that always selects the longer response achieves:

**94.61%**

This is higher than the formal fine-tuned IID pairwise result.

Therefore the 91.35% IID result must not be interpreted as unbiased
semantic preference accuracy.

### Length-Matched Challenge

| Model | Accuracy |
|---|---:|
| Base | 49.56% |
| Fine-tuned | **77.78%** |

### Reversed-Length Challenge

Every preferred response is shorter than its rejected counterpart.

| Model | Accuracy |
|---|---:|
| Base | 47.70% |
| Fine-tuned | **74.90%** |

The fine-tuned model therefore learned meaningful preference signal,
but also relied substantially on shortcut features present in V1.

---

## 7. Truncation audit

The formal configuration used:

`max_length = 512`

A later token-length audit found that approximately:

**98.54% of the 2,255 unique test responses were truncated.**

This is treated as a V1 experimental-design limitation.

The observed correlation between reward and full response token
length was also high:

- Pearson: approximately **0.820**
- quality-level residualized Pearson: approximately **0.579**

---

## 8. Checkpoint-selection audit

The trainer selected step 800 because its evaluation loss was lower:

| Checkpoint | Eval Loss |
|---|---:|
| Step 800 | **0.11339** |
| Step 1000 | 0.13200 |

A later controlled BF16 evaluation on the same 2,255 responses found:

| Checkpoint | Pairwise Accuracy |
|---|---:|
| Step 800 | 92.20% |
| Step 1000 | **92.64%** |

Step 1000 was also slightly stronger on NDCG@5 and perfect 5-way
ranking, while step 800 retained slightly stronger rank-correlation
metrics.

This demonstrates that the checkpoint with the lowest validation loss
is not necessarily the checkpoint with the best downstream ranking
behavior.

---

## 9. Main conclusion

V1 demonstrates that domain preference learning can significantly
adapt an 8B Reward Model, but it also demonstrates why IID accuracy
alone is insufficient.

The most important lesson is:

> Reward Modeling is not only about minimizing a pairwise loss. It is
> about verifying that the learned reward function actually rewards
> the behavior we intend.

The next version will focus on length-balanced data, semantic hard
negatives, human-verified gold evaluation, longer-context ablations,
multi-seed validation, and ranking-aware checkpoint selection.
