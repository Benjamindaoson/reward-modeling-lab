from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .data import iter_preference_records
from .model_artifacts import load_reward_model
from .train import _format_pair


def evaluate_pairwise(model_path: Path, data_path: Path, max_length: int, batch_size: int, base_model_path: Path | None = None) -> dict[str, float | int]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("transformers and torch are required for reward-model evaluation") from error
    tokenizer, model = load_reward_model(model_path, base_model_path)
    records = list(iter_preference_records(data_path))
    correct = 0
    margins: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            def score(answer_key: str) -> Any:
                encoded = tokenizer([_format_pair(str(record["question"]), str(record[answer_key])) for record in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
                return model(**encoded).logits.squeeze(-1).float().cpu()
            chosen, rejected = score("chosen"), score("rejected")
            margins.extend((chosen - rejected).tolist())
            correct += int((chosen > rejected).sum().item())
    return {"records": len(records), "pairwise_accuracy": correct / len(records) if records else 0.0, "mean_reward_margin": sum(margins) / len(margins) if margins else 0.0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate pairwise financial reward-model accuracy")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-model-path", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(evaluate_pairwise(args.model_path, args.data, args.max_length, args.batch_size, args.base_model_path)))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
