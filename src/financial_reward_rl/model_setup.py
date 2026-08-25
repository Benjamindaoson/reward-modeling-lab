"""Resolve a local model directory, downloading it only when the GPU operator requests it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def ensure_model(model_path: Path, model_id: str | None, revision: str | None) -> dict[str, str]:
    config_file = model_path / "config.json"
    if config_file.is_file():
        return {"model_path": str(model_path), "source": "existing-local-model"}
    if not model_id:
        raise FileNotFoundError(f"{config_file} does not exist; provide --model-id to download a base reward model")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to download --model-id") from error
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(model_path))
    if not config_file.is_file():
        raise RuntimeError(f"downloaded model {model_id} but config.json is absent from {model_path}")
    return {"model_path": str(model_path), "source": f"downloaded:{model_id}"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a local base reward-model directory.")
    parser.add_argument("--model-path", default=Path("models/base-reward-model"), type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(ensure_model(args.model_path, args.model_id, args.revision), ensure_ascii=False))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
