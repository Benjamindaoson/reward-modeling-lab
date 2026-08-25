from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path


REQUIRED_GPU_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "pandas",
    "pyarrow",
)


def inspect_gpu_readiness(train_file: Path, eval_file: Path, model_path: Path) -> dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_detail: object = "nvidia-smi not found"
    gpu_available = False
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
            gpu_available = result.returncode == 0 and bool(result.stdout.strip())
            gpu_detail = result.stdout.strip().splitlines() if gpu_available else (result.stderr.strip() or "nvidia-smi returned no GPU")
        except (OSError, subprocess.TimeoutExpired) as error:
            gpu_detail = str(error)
    packages = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_GPU_PACKAGES}
    paths = {"train_file": train_file.is_file(), "eval_file": eval_file.is_file(), "model_path": model_path.exists()}
    blockers = []
    if not gpu_available:
        blockers.append(f"GPU unavailable: {gpu_detail}")
    missing_packages = [name for name, available in packages.items() if not available]
    if missing_packages:
        blockers.append(f"Missing packages: {', '.join(missing_packages)}")
    missing_paths = [name for name, available in paths.items() if not available]
    if missing_paths:
        blockers.append(f"Missing paths: {', '.join(missing_paths)}")
    return {"ready": not blockers, "gpu": {"available": gpu_available, "detail": gpu_detail}, "packages": packages, "paths": paths, "blockers": blockers}
