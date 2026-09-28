"""V2 data construction for shortcut-robust reward modeling.

V2 deliberately reuses existing preference labels. It does not synthesize new
"correct" or "wrong" labels. The anti-shortcut hard negatives are naturally
occurring pairs where the preferred response is shorter than the rejected
response.

The public builder is deterministic, preserves source metadata, and records the
sampling policy in a manifest so a later GPU run can be audited.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from .data import iter_preference_records


class LengthBucket(StrEnum):
    MATCHED = "length_matched"
    SHORTCUT_ALIGNED = "verbose_preferred_concise_rejected"
    ANTI_LENGTH = "concise_preferred_verbose_rejected"


@dataclass(frozen=True, slots=True)
class LengthFeatures:
    chosen_length: int
    rejected_length: int
    relative_gap: float
    bucket: LengthBucket


def _whitespace_length(text: str) -> int:
    return max(1, len(text.split()))


def create_length_counter(
    tokenizer_path: Path | None = None,
) -> tuple[Callable[[str], int], str]:
    """Create an answer-length counter.

    Production V2 should pass the same tokenizer as the reward model.
    Small CI fixtures can omit it and use whitespace-token counts.
    """

    if tokenizer_path is None:
        return _whitespace_length, "whitespace_tokens"

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required when tokenizer_path is supplied"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    def count(text: str) -> int:
        return max(
            1,
            len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            ),
        )

    return count, f"hf_tokenizer:{tokenizer_path}"


def pair_length_features(
    record: dict[str, Any],
    counter: Callable[[str], int],
    *,
    matched_relative_gap: float = 0.10,
) -> LengthFeatures:
    chosen_length = counter(str(record["chosen"]))
    rejected_length = counter(str(record["rejected"]))
    relative_gap = abs(chosen_length - rejected_length) / max(
        chosen_length,
        rejected_length,
        1,
    )

    if relative_gap <= matched_relative_gap:
        bucket = LengthBucket.MATCHED
    elif chosen_length < rejected_length:
        bucket = LengthBucket.ANTI_LENGTH
    else:
        bucket = LengthBucket.SHORTCUT_ALIGNED

    return LengthFeatures(
        chosen_length=chosen_length,
        rejected_length=rejected_length,
        relative_gap=relative_gap,
        bucket=bucket,
    )


def _source_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "question": record["question"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _annotate(
    record: dict[str, Any],
    features: LengthFeatures,
    *,
    sampling_round: int,
) -> dict[str, Any]:
    output = dict(record)
    output["_v2"] = {
        "source_hash": _source_hash(record),
        "length_bucket": features.bucket.value,
        "chosen_length": features.chosen_length,
        "rejected_length": features.rejected_length,
        "relative_length_gap": round(features.relative_gap, 8),
        "sampling_round": sampling_round,
    }
    return output


def _sample_bucket(
    items: list[tuple[dict[str, Any], LengthFeatures]],
    target: int,
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not items:
        raise ValueError("cannot sample an empty length bucket")

    order = list(range(len(items)))
    rng.shuffle(order)
    output: list[dict[str, Any]] = []
    for position in range(target):
        index = order[position % len(order)]
        sampling_round = position // len(order)
        record, features = items[index]
        output.append(
            _annotate(
                record,
                features,
                sampling_round=sampling_round,
            )
        )
    return output


def build_balanced_training_view(
    source_path: Path,
    output_path: Path,
    *,
    counter: Callable[[str], int] = _whitespace_length,
    length_metric: str = "whitespace_tokens",
    matched_relative_gap: float = 0.10,
    max_repeat: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a deterministic three-way length-balanced training view.

    Strata:
    - length-matched preference pairs;
    - shortcut-aligned pairs where the preferred response is longer;
    - anti-length hard negatives where the preferred response is shorter.

    Minority strata can be repeated deterministically, but no source example is
    repeated more than max_repeat times.
    """

    if not 0 <= matched_relative_gap < 1:
        raise ValueError("matched_relative_gap must be in [0, 1)")
    if max_repeat < 1:
        raise ValueError("max_repeat must be >= 1")

    records = list(iter_preference_records(source_path))
    if not records:
        raise ValueError(f"no records found: {source_path}")

    buckets: dict[
        LengthBucket,
        list[tuple[dict[str, Any], LengthFeatures]],
    ] = {bucket: [] for bucket in LengthBucket}

    for record in records:
        features = pair_length_features(
            record,
            counter,
            matched_relative_gap=matched_relative_gap,
        )
        buckets[features.bucket].append((record, features))

    counts_before = {
        bucket.value: len(items)
        for bucket, items in buckets.items()
    }
    if any(count == 0 for count in counts_before.values()):
        missing = [
            name
            for name, count in counts_before.items()
            if count == 0
        ]
        raise ValueError(
            "V2 balancing requires all three length strata; missing: "
            + ", ".join(missing)
        )

    smallest = min(counts_before.values())
    target_per_bucket = min(
        max(counts_before.values()),
        smallest * max_repeat,
    )
    if target_per_bucket <= 0:
        raise ValueError("V2 balancing produced an empty target")

    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for bucket in (
        LengthBucket.MATCHED,
        LengthBucket.SHORTCUT_ALIGNED,
        LengthBucket.ANTI_LENGTH,
    ):
        balanced.extend(
            _sample_bucket(
                buckets[bucket],
                target_per_bucket,
                rng=rng,
            )
        )

    rng.shuffle(balanced)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in balanced:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts_after = {bucket.value: 0 for bucket in LengthBucket}
    repeated_examples = 0
    for record in balanced:
        meta = record["_v2"]
        counts_after[meta["length_bucket"]] += 1
        repeated_examples += int(meta["sampling_round"] > 0)

    return {
        "source": str(source_path),
        "output": str(output_path),
        "seed": seed,
        "length_metric": length_metric,
        "matched_relative_gap": matched_relative_gap,
        "max_repeat": max_repeat,
        "source_pairs": len(records),
        "output_pairs": len(balanced),
        "target_per_bucket": target_per_bucket,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "anti_length_fraction_before": (
            counts_before[LengthBucket.ANTI_LENGTH.value] / len(records)
        ),
        "anti_length_fraction_after": (
            counts_after[LengthBucket.ANTI_LENGTH.value] / len(balanced)
        ),
        "shortcut_aligned_fraction_before": (
            counts_before[LengthBucket.SHORTCUT_ALIGNED.value] / len(records)
        ),
        "shortcut_aligned_fraction_after": (
            counts_after[LengthBucket.SHORTCUT_ALIGNED.value] / len(balanced)
        ),
        "repeated_output_instances": repeated_examples,
    }


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_challenge_views(
    test_path: Path,
    output_dir: Path,
    *,
    counter: Callable[[str], int] = _whitespace_length,
    length_metric: str = "whitespace_tokens",
    matched_relative_gap: float = 0.10,
) -> dict[str, Any]:
    """Materialize fixed IID, matched-length and reversed-length test views."""

    records = list(iter_preference_records(test_path))
    matched: list[dict[str, Any]] = []
    reversed_length: list[dict[str, Any]] = []

    for record in records:
        features = pair_length_features(
            record,
            counter,
            matched_relative_gap=matched_relative_gap,
        )
        annotated = _annotate(record, features, sampling_round=0)
        if features.bucket == LengthBucket.MATCHED:
            matched.append(annotated)
        elif features.bucket == LengthBucket.ANTI_LENGTH:
            reversed_length.append(annotated)

    paths = {
        "iid": output_dir / "iid.jsonl",
        "length_matched": output_dir / "length_matched.jsonl",
        "reversed_length": output_dir / "reversed_length.jsonl",
    }
    _write_records(paths["iid"], records)
    _write_records(paths["length_matched"], matched)
    _write_records(paths["reversed_length"], reversed_length)

    return {
        "source": str(test_path),
        "length_metric": length_metric,
        "matched_relative_gap": matched_relative_gap,
        "counts": {
            "iid": len(records),
            "length_matched": len(matched),
            "reversed_length": len(reversed_length),
        },
        "paths": {name: str(path) for name, path in paths.items()},
    }


def build_eval_monitor(
    eval_path: Path,
    output_path: Path,
    *,
    size: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    records = list(iter_preference_records(eval_path))
    if not records:
        raise ValueError(f"no records found: {eval_path}")
    rng = random.Random(seed)
    selected = records if len(records) <= size else rng.sample(records, size)
    _write_records(output_path, selected)
    return {
        "source": str(eval_path),
        "output": str(output_path),
        "source_pairs": len(records),
        "monitor_pairs": len(selected),
        "seed": seed,
    }


def audit_truncation(
    path: Path,
    counter: Callable[[str], int],
    *,
    context_lengths: tuple[int, ...] = (512, 1024),
) -> dict[str, Any]:
    """Audit answer-only truncation pressure over unique responses."""

    records = list(iter_preference_records(path))
    unique_responses = {
        str(record[key])
        for record in records
        for key in ("chosen", "rejected")
    }
    lengths = [counter(text) for text in unique_responses]
    result: dict[str, Any] = {
        "unique_responses": len(unique_responses),
        "mean_length": (
            sum(lengths) / len(lengths)
            if lengths
            else 0.0
        ),
        "max_length": max(lengths) if lengths else 0,
        "contexts": {},
    }
    for context in context_lengths:
        truncated = sum(length > context for length in lengths)
        result["contexts"][str(context)] = {
            "truncated_responses": truncated,
            "truncation_rate": (
                truncated / len(lengths)
                if lengths
                else 0.0
            ),
        }
    return result


def prepare_v2_bundle(
    *,
    train_path: Path,
    eval_path: Path,
    test_path: Path,
    output_dir: Path,
    tokenizer_path: Path | None = None,
    matched_relative_gap: float = 0.10,
    max_repeat: int = 4,
    monitor_size: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    counter, length_metric = create_length_counter(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = build_balanced_training_view(
        train_path,
        output_dir / "train_balanced_hard.jsonl",
        counter=counter,
        length_metric=length_metric,
        matched_relative_gap=matched_relative_gap,
        max_repeat=max_repeat,
        seed=seed,
    )
    monitor_summary = build_eval_monitor(
        eval_path,
        output_dir / f"eval_monitor_{monitor_size}.jsonl",
        size=monitor_size,
        seed=seed,
    )
    challenge_summary = build_challenge_views(
        test_path,
        output_dir / "challenges",
        counter=counter,
        length_metric=length_metric,
        matched_relative_gap=matched_relative_gap,
    )
    truncation_summary = audit_truncation(
        test_path,
        counter,
        context_lengths=(512, 1024),
    )

    manifest = {
        "schema_version": "2.0",
        "purpose": "shortcut_robust_reward_model_v2",
        "policy": {
            "new_labels_synthesized": False,
            "anti_length_hard_negative_definition": (
                "existing preference pair with preferred/chosen response "
                "shorter than rejected response"
            ),
            "balanced_strata": [
                bucket.value for bucket in LengthBucket
            ],
        },
        "train": train_summary,
        "monitor": monitor_summary,
        "challenges": challenge_summary,
        "truncation_audit": truncation_summary,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = str(manifest_path)
    return manifest
