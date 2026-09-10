<p align="center">
  <img src="./assets/hero.svg" alt="Reward Modeling Lab — auditable 8B post-training" width="100%" />
</p>

<p align="center">
  <a href="../README.md"><b>Full Research README</b></a> ·
  <a href="../docs/results/"><b>Results</b></a> ·
  <a href="../configs/training/"><b>Training Configs</b></a> ·
  <a href="../tests/"><b>Tests</b></a>
</p>

---

## The result is not the point. The audit is.

The initial fine-tuned reward model reached **91.35%** frozen held-out pairwise accuracy. But a trivial **“prefer the longer response”** heuristic reached **94.61%** on the original V1 test distribution.

That changed the project from a training demo into a reward-function investigation:

<table>
<tr>
<td width="33%" valign="top">

### 01 · Train

**Skywork Reward Llama 3.1 8B**

Single **NVIDIA A10 23GB** · 4-bit NF4 QLoRA · BF16 · **1,000 optimizer steps**.

</td>
<td width="33%" valign="top">

### 02 · Attack

**Challenge the shortcut**

Length-matched, reversed-length, truncation, reward-length correlation and checkpoint tests.

</td>
<td width="33%" valign="top">

### 03 · Verify

**Measure ranking behavior**

Kendall **τ = 0.8828** · NDCG@5 **= 0.9474** · controlled pairwise challenge sets.

</td>
</tr>
</table>

---

## Results that survived deeper inspection

| Evaluation | Base | Fine-tuned / measured |
|---|---:|---:|
| **Frozen held-out pairwise** | 50.42% | **91.35%** |
| **Length-matched challenge** | 49.56% | **77.78%** |
| **Reversed-length challenge** | 47.70% | **74.90%** |
| **Kendall τ · 5-way ranking** | — | **0.8828** |
| **NDCG@5 · 5-way ranking** | — | **0.9474** |

> **Interpretation:** the model clearly learned useful preference signal, but the original distribution also contained a strong exploitable length shortcut. The project therefore reports both the improvement **and** the failure mode.

---

## Research loop

```mermaid
flowchart LR
    A[Preference Data] --> B[8B QLoRA Training]
    B --> C[Frozen Eval]
    C --> D[Shortcut Attack]
    D --> E[Controlled Challenges]
    E --> F[Ranking Audit]
    F --> G[V2 Data / Eval Redesign]
```

**Train → Attack → Diagnose → Redesign** is the organizing principle of the repository.

---

## Evidence bundle

| What is actually completed | Evidence scope |
|---|---|
| **Real GPU training** | 8B model · single A10 23GB · 1,000 optimizer steps |
| **Preference data** | 35,990 available training pairs · formal run processed ~8,000 pair instances |
| **Frozen test** | 451 questions · 4,510 comparisons · 2,255 unique responses |
| **Shortcut audit** | length heuristic · matched-length · reversed-length challenges |
| **Ranking audit** | Kendall τ · NDCG@5 · perfect 5-way ranking |
| **Context audit** | truncation behavior · reward/full-token-length correlation |
| **Checkpoint audit** | step-800 vs step-1000 behavior |
| **Reproducibility** | versioned configs · machine-readable results · CI / tests |

<p align="center">
  <img src="https://img.shields.io/badge/8B-Reward%20Model-7C3AED?style=for-the-badge" />
  <img src="https://img.shields.io/badge/4--bit-QLoRA-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Single%20GPU-A10%2023GB-4F46E5?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Focus-Reward%20Audit-A855F7?style=for-the-badge" />
</p>

---

## Architecture

```mermaid
flowchart LR
    A[Preference Archive] --> B[Schema / Split Validation]
    B --> C[QLoRA Training]
    C --> D[Checkpoints]
    D --> E[Pairwise Eval]
    D --> F[Ranking Eval]
    D --> G[Shortcut Challenges]
    D --> H[Context / Length Audit]
    E & F & G & H --> I[JSON / CSV / Figures]
    I --> J[Auditable Evidence]
```

---

## Reproduce the formal run

The frozen training configuration is:

[`configs/training/formal_gpu_qlora_1000_final.json`](../configs/training/formal_gpu_qlora_1000_final.json)

Use the **[full research README](../README.md)** for environment setup, data preparation, exact training/evaluation commands and artifact policy.

<details>
<summary><b>Explicitly not claimed as completed</b></summary>
<br/>

`GRPO` · `PPO` · multi-node training · multi-GPU distributed training · production RM serving benchmarks · online RL deployment

The repository deliberately separates **executed experiments** from future work.

</details>

---

<p align="center">
  <b>A reward model is useful only if its reward survives adversarial inspection.</b>
</p>
