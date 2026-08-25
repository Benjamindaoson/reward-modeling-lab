"""Load full reward checkpoints and QLoRA adapters through one explicit contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_model_artifact(model_path: Path, base_model_path: Path | None = None) -> dict[str, Any]:
    adapter_path = model_path / "adapter_config.json"
    if adapter_path.is_file():
        if base_model_path is None:
            raise ValueError("this is a QLoRA adapter; provide --base-model-path")
        if not (base_model_path / "config.json").is_file():
            raise FileNotFoundError(f"base model config not found: {base_model_path / 'config.json'}")
        return {
            "is_adapter": True,
            "base_model_path": base_model_path,
            "adapter_path": model_path,
            "load_in_4bit": True,
        }
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model_path / 'config.json'}")
    return {"is_adapter": False, "base_model_path": model_path, "adapter_path": None, "load_in_4bit": False}


def load_reward_model(model_path: Path, base_model_path: Path | None = None) -> tuple[Any, Any]:
    """Load a full classifier or a 4-bit base model with its trained QLoRA adapter."""
    plan = resolve_model_artifact(model_path, base_model_path)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("torch and transformers are required to load a reward model") from error

    tokenizer = AutoTokenizer.from_pretrained(plan["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not plan["is_adapter"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            plan["base_model_path"], trust_remote_code=True, device_map="auto"
        )
    else:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("peft is required to load a QLoRA reward adapter") from error
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            plan["base_model_path"],
            trust_remote_code=True,
            quantization_config=quantization,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, plan["adapter_path"])
    model.eval()
    return tokenizer, model
