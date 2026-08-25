"""CPU-safe commands for preparing and validating the reward-model project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import extract_preference_data_splits, summarize_preference_data
from .preflight import inspect_gpu_readiness


def main() -> None:
    parser = argparse.ArgumentParser(prog="financial-reward-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Extract preference-data JSONL splits from an archive.")
    prepare.add_argument("--archive", required=True, type=Path)
    prepare.add_argument("--output-dir", default=Path("data/preferences"), type=Path)

    summarize = subparsers.add_parser("summarize-data", help="Validate and summarize one preference JSONL file.")
    summarize.add_argument("--data", required=True, type=Path)

    preflight = subparsers.add_parser("preflight", help="Check GPU, packages and input paths before training.")
    preflight.add_argument("--train-file", required=True, type=Path)
    preflight.add_argument("--eval-file", required=True, type=Path)
    preflight.add_argument("--model-path", required=True, type=Path)
    preflight.add_argument("--require-ready", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare-data":
        result = extract_preference_data_splits(args.archive, args.output_dir)
    elif args.command == "summarize-data":
        result = summarize_preference_data(args.data)
    else:
        result = inspect_gpu_readiness(args.train_file, args.eval_file, args.model_path)
        if args.require_ready and not result["ready"]:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            raise SystemExit("GPU training preflight failed")

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
