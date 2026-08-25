from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .data import iter_preference_records
from .model_artifacts import load_reward_model
from .train import _format_pair


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def summarize_scores(
    chosen: list[float],
    rejected: list[float],
) -> dict[str, Any]:
    if not chosen:
        return {
            "records": 0,
            "pairwise_accuracy": None,
            "mean_reward_margin": None,
            "mean_chosen_reward": None,
            "mean_rejected_reward": None,
        }

    margins = [
        c - r
        for c, r in zip(chosen, rejected)
    ]

    correct = sum(
        1
        for margin in margins
        if margin > 0
    )

    return {
        "records": len(margins),

        "pairwise_accuracy": (
            correct / len(margins)
        ),

        "mean_reward_margin": statistics.mean(
            margins
        ),

        "mean_chosen_reward": statistics.mean(
            chosen
        ),

        "mean_rejected_reward": statistics.mean(
            rejected
        ),

        "margin_std": (
            statistics.pstdev(margins)
            if len(margins) > 1
            else 0.0
        ),

        "margin_p10": percentile(
            margins,
            0.10,
        ),

        "margin_p50": percentile(
            margins,
            0.50,
        ),

        "margin_p90": percentile(
            margins,
            0.90,
        ),

        "chosen_reward_std": (
            statistics.pstdev(chosen)
            if len(chosen) > 1
            else 0.0
        ),

        "rejected_reward_std": (
            statistics.pstdev(rejected)
            if len(rejected) > 1
            else 0.0
        ),
    }


def update_manifest(
    run_dir: Path,
    phase: str,
) -> None:
    path = run_dir / "manifest.json"

    if not path.is_file():
        return

    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    pipeline = manifest.setdefault(
        "pipeline",
        {},
    )

    plan = manifest.setdefault(
        "experiment_plan",
        {},
    )

    # MONITOR_PHASES_V1
    if phase == "base":
        pipeline["base_evaluation"] = "completed"

        plan["08_base_vs_finetuned"] = (
            "base_available"
        )

        plan["09_quality_gap_analysis"] = (
            "base_available"
        )

    elif phase == "base_monitor":
        pipeline["base_monitor_evaluation"] = "completed"
        plan["06_formal_training"] = "monitor_baseline_available"

    elif phase == "finetuned_monitor":
        pipeline["finetuned_monitor_evaluation"] = "completed"

    elif phase == "finetuned":
        pipeline[
            "finetuned_evaluation"
        ] = "completed"

        pipeline[
            "quality_gap_evaluation"
        ] = "completed"

        plan[
            "08_base_vs_finetuned"
        ] = "data_available"

        plan[
            "09_quality_gap_analysis"
        ] = "data_available"

    manifest["status"] = "in_progress"
    manifest["updated_at"] = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def evaluate(
    *,
    model_path: Path,
    data_path: Path,
    max_length: int,
    batch_size: int,
    base_model_path: Path | None,
    run_dir: Path,
    phase: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required"
        ) from exc

    records = list(
        iter_preference_records(
            data_path
        )
    )

    if not records:
        raise ValueError(
            f"no records found: {data_path}"
        )

    print(
        json.dumps(
            {
                "event": "evaluation_start",
                "phase": phase,
                "records": len(records),
                "batch_size": batch_size,
                "max_length": max_length,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    total_start = time.time()

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------
    model_load_start = time.time()

    tokenizer, model = load_reward_model(
        model_path,
        base_model_path,
    )

    model.eval()

    model_load_seconds = (
        time.time() - model_load_start
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Score storage
    # --------------------------------------------------------
    overall_chosen: list[float] = []
    overall_rejected: list[float] = []

    grouped: dict[
        str,
        dict[str, list[float]],
    ] = {}

    total_batches = math.ceil(
        len(records) / batch_size
    )

    progress_interval = max(
        1,
        total_batches // 20,
    )

    eval_start = time.time()

    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(
                0,
                len(records),
                batch_size,
            ),
            start=1,
        ):
            batch = records[
                start:start + batch_size
            ]

            def score(
                answer_key: str,
            ) -> Any:
                texts = [
                    _format_pair(
                        str(record["question"]),
                        str(record[answer_key]),
                    )
                    for record in batch
                ]

                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(model.device)

                return (
                    model(**encoded)
                    .logits
                    .squeeze(-1)
                    .float()
                    .cpu()
                    .tolist()
                )

            chosen = score("chosen")
            rejected = score("rejected")

            overall_chosen.extend(chosen)
            overall_rejected.extend(
                rejected
            )

            for record, c, r in zip(
                batch,
                chosen,
                rejected,
            ):
                gap = str(
                    record.get(
                        "quality_gap",
                        "unknown",
                    )
                )

                bucket = grouped.setdefault(
                    gap,
                    {
                        "chosen": [],
                        "rejected": [],
                    },
                )

                bucket["chosen"].append(
                    float(c)
                )
                bucket["rejected"].append(
                    float(r)
                )

            if (
                batch_index == 1
                or batch_index
                % progress_interval
                == 0
                or batch_index
                == total_batches
            ):
                elapsed = (
                    time.time()
                    - eval_start
                )

                progress = (
                    batch_index
                    / total_batches
                )

                eta = (
                    elapsed
                    / progress
                    - elapsed
                    if progress > 0
                    else 0.0
                )

                print(
                    json.dumps(
                        {
                            "event": (
                                "evaluation_progress"
                            ),
                            "phase": phase,
                            "batch": batch_index,
                            "total_batches": (
                                total_batches
                            ),
                            "progress_pct": (
                                100.0
                                * progress
                            ),
                            "elapsed_seconds": (
                                elapsed
                            ),
                            "eta_seconds": eta,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    eval_runtime = (
        time.time()
        - eval_start
    )

    total_runtime = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------
    overall = summarize_scores(
        overall_chosen,
        overall_rejected,
    )

    quality_gap = {
        gap: summarize_scores(
            values["chosen"],
            values["rejected"],
        )
        for gap, values
        in sorted(
            grouped.items(),
            key=lambda item: item[0],
        )
    }

    gpu_name = None
    peak_allocated_gb = None
    peak_reserved_gb = None

    if torch.cuda.is_available():
        gpu_name = (
            torch.cuda
            .get_device_name(0)
        )

        peak_allocated_gb = (
            torch.cuda
            .max_memory_allocated(0)
            / (1024 ** 3)
        )

        peak_reserved_gb = (
            torch.cuda
            .max_memory_reserved(0)
            / (1024 ** 3)
        )

    result = {
        "schema_version": "1.0",

        "phase": phase,

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "model_path": str(
            model_path
        ),

        "base_model_path": (
            str(base_model_path)
            if base_model_path
            else None
        ),

        "data_path": str(
            data_path
        ),

        "records": len(
            records
        ),

        "batch_size": (
            batch_size
        ),

        "max_length": (
            max_length
        ),

        "model_load_seconds": (
            model_load_seconds
        ),

        "eval_runtime_seconds": (
            eval_runtime
        ),

        "total_runtime_seconds": (
            total_runtime
        ),

        "records_per_second": (
            len(records)
            / eval_runtime
            if eval_runtime > 0
            else None
        ),

        "gpu_name": (
            gpu_name
        ),

        "peak_gpu_memory_allocated_gb": (
            peak_allocated_gb
        ),

        "peak_gpu_memory_reserved_gb": (
            peak_reserved_gb
        ),

        "overall": (
            overall
        ),

        "quality_gap": (
            quality_gap
        ),
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------
    data_dir = (
        run_dir
        / "data"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        data_dir
        / f"{phase}_evaluation.json"
    )

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save quality-gap CSV
    # --------------------------------------------------------
    csv_path = (
        data_dir
        / f"{phase}_quality_gap.csv"
    )

    columns = [
        "quality_gap",
        "records",
        "pairwise_accuracy",
        "mean_reward_margin",
        "mean_chosen_reward",
        "mean_rejected_reward",
        "margin_std",
        "margin_p10",
        "margin_p50",
        "margin_p90",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for gap, metrics in (
            quality_gap.items()
        ):
            row = {
                "quality_gap": gap,
            }

            for column in columns[1:]:
                row[column] = (
                    metrics.get(column)
                )

            writer.writerow(row)

    update_manifest(
        run_dir,
        phase,
    )

    print(
        json.dumps(
            {
                "event": (
                    "evaluation_complete"
                ),
                "phase": phase,
                "json": str(
                    json_path
                ),
                "quality_gap_csv": str(
                    csv_path
                ),
                "overall": overall,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    return result


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Financial Reward Model "
            "evaluation suite."
        )
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--base-model-path",
        type=Path,
    )

    parser.add_argument(
        "--data",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "base",
            "finetuned",
            "base_monitor",
            "finetuned_monitor",
        ],
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    args = parser.parse_args(
        argv
    )

    evaluate(
        model_path=args.model_path,
        data_path=args.data,
        max_length=args.max_length,
        batch_size=args.batch_size,
        base_model_path=(
            args.base_model_path
        ),
        run_dir=args.run_dir,
        phase=args.phase,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
