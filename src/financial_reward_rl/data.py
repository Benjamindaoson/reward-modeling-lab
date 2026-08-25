from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator


PREFERENCE_DATA_MEMBERS = {
    "train": "reward_model_data/reward_data/train/preference_dataset.jsonl",
    "eval": "reward_model_data/reward_data/eval/preference_dataset.jsonl",
    "test": "reward_model_data/reward_data/test/preference_dataset.jsonl",
}
REQUIRED_FIELDS = ("question", "chosen", "rejected")


def extract_preference_data_splits(archive: Path, output_dir: Path) -> dict[str, Path]:
    """Extract the required pairwise preference splits from an archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as source:
        for split, member in PREFERENCE_DATA_MEMBERS.items():
            try:
                content = source.read(member)
            except KeyError as error:
                raise ValueError(f"preference data archive is missing {member}") from error
            output_path = output_dir / f"{split}.jsonl"
            with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
            temporary_path.replace(output_path)
            extracted[split] = output_path
    return extracted


def iter_preference_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict) or not all(isinstance(record.get(field), str) and record[field].strip() for field in REQUIRED_FIELDS):
                raise ValueError(f"missing non-empty pairwise fields at {path}:{line_number}")
            yield record


def summarize_preference_data(path: Path) -> dict[str, object]:
    total = 0
    quality_gaps: dict[str, int] = {}
    for record in iter_preference_records(path):
        total += 1
        gap = str(record.get("quality_gap", "unknown"))
        quality_gaps[gap] = quality_gaps.get(gap, 0) + 1
    return {"path": str(path), "records": total, "quality_gap_distribution": quality_gaps}
