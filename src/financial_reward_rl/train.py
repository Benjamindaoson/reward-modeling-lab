from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .data import iter_preference_records
from .preflight import inspect_gpu_readiness


def _format_pair(question: str, answer: str) -> str:
    return f"<|user|>\n{question}\n<|assistant|>\n{answer}"


def build_model_load_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Return a dependency-free loading plan for the multi-GPU and single-GPU profiles."""
    mode = config.get("training_mode", "freeze_bf16")
    if mode == "freeze_bf16":
        return {"load_in_4bit": False, "use_freeze_tuning": True}
    if mode != "qlora_4bit":
        raise ValueError(f"unsupported training_mode: {mode}")
    return {
        "load_in_4bit": True,
        "quantization_config": {
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": config["torch_dtype"],
        },
        "lora": {
            "r": int(config["lora_rank"]),
            "lora_alpha": int(config["lora_alpha"]),
            "lora_dropout": float(config["lora_dropout"]),
            "target_modules": list(config["lora_target_modules"]),
            "modules_to_save": list(config.get("lora_modules_to_save", ["score"])),
        },
        "use_freeze_tuning": False,
    }


def setup_freeze_tuning(model: Any, layers: int, modules: str, extra_modules: str | None) -> dict[str, int]:
    """Configure trainable parameters for reward-model adaptation."""
    total_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if total_layers <= 0:
        raise ValueError("the selected reward model does not expose num_hidden_layers")
    trainable_layer_ids = range(max(0, total_layers - layers), total_layers)
    patterns: list[str] = []
    for layer_id in trainable_layer_ids:
        if modules == "all":
            patterns.append(f".{layer_id}.")
        else:
            patterns.extend(f".{layer_id}.{module.strip()}" for module in modules.split(",") if module.strip())
    if extra_modules:
        patterns.extend(module.strip() for module in extra_modules.split(",") if module.strip())

    trainable_parameters = total_parameters = 0
    for name, parameter in model.named_parameters():
        total_parameters += parameter.numel()
        parameter.requires_grad = any(pattern in name for pattern in patterns)
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
    if not trainable_parameters:
        raise ValueError("freeze-tuning patterns matched no model parameters")
    return {"total_layers": total_layers, "trainable_parameters": trainable_parameters, "total_parameters": total_parameters}


def run_training(config_path: Path, model_path: Path, resume_from_checkpoint: str | None = None) -> None:
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    train_file = Path(config["train_file"])
    eval_file = Path(config["eval_file"])
    readiness = inspect_gpu_readiness(train_file, eval_file, model_path)
    if not readiness["ready"]:
        raise RuntimeError("reward-model launch blocked: " + "; ".join(readiness["blockers"]))
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, Trainer, TrainingArguments
    except ImportError as error:
        raise RuntimeError("GPU training packages are unavailable") from error

    class PreferenceDataset(Dataset):
        def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
            self.records = list(iter_preference_records(path))
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            chosen = self.tokenizer(_format_pair(str(record["question"]), str(record["chosen"])), truncation=True, max_length=self.max_length)
            rejected = self.tokenizer(_format_pair(str(record["question"]), str(record["rejected"])), truncation=True, max_length=self.max_length)
            return {"chosen_input_ids": chosen["input_ids"], "chosen_attention_mask": chosen["attention_mask"], "rejected_input_ids": rejected["input_ids"], "rejected_attention_mask": rejected["attention_mask"]}

    class PairwiseCollator:
        def __init__(self, tokenizer: Any) -> None:
            self.tokenizer = tokenizer

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            def pad(prefix: str) -> dict[str, Any]:
                return self.tokenizer.pad([{"input_ids": feature[f"{prefix}_input_ids"], "attention_mask": feature[f"{prefix}_attention_mask"]} for feature in features], padding=True, return_tensors="pt")
            chosen, rejected = pad("chosen"), pad("rejected")
            return {"chosen_input_ids": chosen["input_ids"], "chosen_attention_mask": chosen["attention_mask"], "rejected_input_ids": rejected["input_ids"], "rejected_attention_mask": rejected["attention_mask"]}

    # PAIRWISE_METRICS_V1
    class PairwiseRewardTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            chosen_rewards = model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
            ).logits.squeeze(-1)

            rejected_rewards = model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
            ).logits.squeeze(-1)

            margins = chosen_rewards - rejected_rewards
            loss = -torch.nn.functional.logsigmoid(margins).mean()

            # 只累计训练阶段的 reward 统计。
            # evaluation 阶段由 compute_metrics 单独统计。
            if model.training:
                stats = getattr(
                    self,
                    "_pairwise_train_stats",
                    {
                        "chosen_sum": 0.0,
                        "rejected_sum": 0.0,
                        "margin_sum": 0.0,
                        "correct": 0.0,
                        "count": 0,
                    },
                )

                with torch.no_grad():
                    chosen = chosen_rewards.detach().float()
                    rejected = rejected_rewards.detach().float()
                    margin = margins.detach().float()

                    stats["chosen_sum"] += chosen.sum().item()
                    stats["rejected_sum"] += rejected.sum().item()
                    stats["margin_sum"] += margin.sum().item()
                    stats["correct"] += (margin > 0).float().sum().item()
                    stats["count"] += chosen.numel()

                self._pairwise_train_stats = stats

            outputs = {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            }

            return (loss, outputs) if return_outputs else loss

        def prediction_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: Any = None,
        ) -> Any:
            # 强制 evaluation 也走 pairwise reward loss，
            # 避免 Trainer 把 chosen/rejected inputs 直接传给基础模型。
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                loss, outputs = self.compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                )

            loss = loss.detach()

            if prediction_loss_only:
                return loss, None, None

            predictions = (
                outputs["chosen_rewards"].detach(),
                outputs["rejected_rewards"].detach(),
            )

            # Trainer 只有同时拿到 predictions 和 label_ids 时
            # 才会稳定调用 compute_metrics。
            # Pairwise RM 本身不需要传统 labels，因此放一个 dummy label。
            labels = torch.zeros_like(
                outputs["chosen_rewards"].detach(),
                dtype=torch.long,
            )

            return loss, predictions, labels

        def log(self, logs: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            logs = dict(logs)

            # EXPERIMENT_TIMING_V1
            now = time.time()

            if not hasattr(self, "_experiment_start_time"):
                self._experiment_start_time = now

            wall_elapsed = now - self._experiment_start_time
            step = int(self.state.global_step)

            max_steps = int(
                getattr(self.state, "max_steps", 0)
                or getattr(self.args, "max_steps", 0)
                or 0
            )

            progress_pct = (
                100.0 * step / max_steps
                if max_steps > 0
                else None
            )

            avg_seconds_per_step = (
                wall_elapsed / step
                if step > 0
                else None
            )

            eta_seconds = (
                avg_seconds_per_step * max(0, max_steps - step)
                if avg_seconds_per_step is not None and max_steps > 0
                else None
            )

            if "eval_loss" in logs:
                event_type = "eval"
            elif "loss" in logs:
                event_type = "train"
            elif "train_runtime" in logs:
                event_type = "train_summary"
            else:
                event_type = "other"

            logs["event_type"] = event_type
            logs["timestamp"] = datetime.now().astimezone().isoformat()
            logs["max_steps"] = max_steps
            logs["wall_elapsed_seconds"] = wall_elapsed
            logs["wall_elapsed_hms"] = time.strftime(
                "%H:%M:%S",
                time.gmtime(wall_elapsed),
            )

            if progress_pct is not None:
                logs["progress_pct"] = progress_pct

            if avg_seconds_per_step is not None:
                logs["avg_seconds_per_step"] = avg_seconds_per_step

            if eta_seconds is not None:
                logs["eta_seconds"] = eta_seconds
                logs["eta_hms"] = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(max(0.0, eta_seconds)),
                )

            # Hugging Face 每 logging_steps 调用一次 log。
            # 在这里把区间内累计的 reward 统计一起写进去。
            stats = getattr(self, "_pairwise_train_stats", None)

            if "loss" in logs and stats and stats["count"] > 0:
                count = float(stats["count"])

                logs["train_chosen_reward"] = stats["chosen_sum"] / count
                logs["train_rejected_reward"] = stats["rejected_sum"] / count
                logs["train_reward_margin"] = stats["margin_sum"] / count
                logs["train_pairwise_accuracy"] = stats["correct"] / count

                self._pairwise_train_stats = {
                    "chosen_sum": 0.0,
                    "rejected_sum": 0.0,
                    "margin_sum": 0.0,
                    "correct": 0.0,
                    "count": 0,
                }

            super().log(logs, *args, **kwargs)

            # 每次日志事件永久写入 JSONL。
            # 即使训练中途崩溃，之前的数据仍然存在。
            metrics_path = Path(self.args.output_dir) / "metrics.jsonl"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "step": int(self.state.global_step),
                "epoch": self.state.epoch,
                **logs,
            }

            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, default=float)
                    + "\n"
                )


    def compute_pairwise_metrics(eval_prediction: Any) -> dict[str, float]:
        import numpy as np

        predictions = eval_prediction.predictions

        if not isinstance(predictions, tuple) or len(predictions) < 2:
            raise ValueError(
                f"expected chosen/rejected prediction tuple, got {type(predictions)}"
            )

        chosen = np.asarray(predictions[0]).reshape(-1)
        rejected = np.asarray(predictions[1]).reshape(-1)

        margins = chosen - rejected

        return {
            "pairwise_accuracy": float(np.mean(margins > 0)),
            "mean_reward_margin": float(np.mean(margins)),
            "mean_chosen_reward": float(np.mean(chosen)),
            "mean_rejected_reward": float(np.mean(rejected)),
        }


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_plan = build_model_load_plan(config)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=config["trust_remote_code"])
    model_config.num_labels = 1
    model_config.pad_token_id = tokenizer.pad_token_id
    load_kwargs: dict[str, Any] = {"config": model_config, "trust_remote_code": config["trust_remote_code"]}
    if load_plan["load_in_4bit"]:
        quantization = load_plan["quantization_config"]
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quantization["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, quantization["bnb_4bit_compute_dtype"]),
        )
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["torch_dtype"] = getattr(torch, config["torch_dtype"])
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **load_kwargs)
    if load_plan["use_freeze_tuning"]:
        freeze_summary = setup_freeze_tuning(
            model,
            layers=int(config["freeze_trainable_layers"]),
            modules=str(config["freeze_trainable_modules"]),
            extra_modules=config.get("freeze_extra_modules"),
        )
        print(json.dumps({"freeze_tuning": freeze_summary}, ensure_ascii=False))
    else:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as error:
            raise RuntimeError("QLoRA requires peft and bitsandbytes") from error
        lora = load_plan["lora"]
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", False)))
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora["r"],
                lora_alpha=lora["lora_alpha"],
                lora_dropout=lora["lora_dropout"],
                target_modules=lora["target_modules"],
                modules_to_save=lora["modules_to_save"],
            ),
        )
        print(json.dumps({"qlora": lora}, ensure_ascii=False))
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    train_dataset = PreferenceDataset(train_file, tokenizer, config["max_length"])
    eval_dataset = PreferenceDataset(eval_file, tokenizer, config["max_length"])
    arguments = TrainingArguments(
        output_dir=config["output_dir"], learning_rate=config["learning_rate"], num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"], max_grad_norm=config["max_grad_norm"], seed=config["seed"],
        per_device_train_batch_size=config["per_device_train_batch_size"], per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"], bf16=config["bf16"], logging_steps=config["logging_steps"],
        eval_strategy="steps", eval_steps=config["eval_steps"], save_strategy="steps", save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"], load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        max_steps=int(config.get("max_steps", -1)), tf32=config["tf32"], remove_unused_columns=False, report_to=[],
    )
    trainer = PairwiseRewardTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairwiseCollator(tokenizer),
        compute_metrics=compute_pairwise_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"]))],
    )
    # EXPERIMENT_SUMMARY_V1
    run_started_at = datetime.now().astimezone()
    wall_start = time.time()

    trainer._experiment_start_time = wall_start

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    wall_end = time.time()
    run_finished_at = datetime.now().astimezone()

    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    world_size = max(
        1,
        int(getattr(trainer.args, "world_size", 1)),
    )

    effective_batch_size = (
        int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
        * world_size
    )

    global_step = int(trainer.state.global_step)

    peak_allocated_gb = None
    peak_reserved_gb = None
    gpu_name = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        )
        peak_reserved_gb = (
            torch.cuda.max_memory_reserved(0) / (1024 ** 3)
        )

    trainer_metrics = {}

    for key, value in train_result.metrics.items():
        if isinstance(value, (int, float)):
            trainer_metrics[key] = float(value)
        else:
            trainer_metrics[key] = value

    summary = {
        "status": "completed",
        "started_at": run_started_at.isoformat(),
        "finished_at": run_finished_at.isoformat(),

        "wall_clock_seconds": wall_end - wall_start,
        "wall_clock_hms": time.strftime(
            "%H:%M:%S",
            time.gmtime(wall_end - wall_start),
        ),

        "global_step": global_step,
        "max_steps": int(trainer.state.max_steps),

        "train_records": len(train_dataset),
        "eval_records": len(eval_dataset),

        "per_device_train_batch_size": int(
            config["per_device_train_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,

        "estimated_pairs_processed": (
            global_step * effective_batch_size
        ),

        "max_length": int(config["max_length"]),
        "learning_rate": float(config["learning_rate"]),
        "training_mode": config.get("training_mode"),

        "model_path": str(model_path),
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "output_dir": str(config["output_dir"]),

        "gpu_name": gpu_name,
        "peak_gpu_memory_allocated_gb": peak_allocated_gb,
        "peak_gpu_memory_reserved_gb": peak_reserved_gb,

        "trainer_metrics": trainer_metrics,
        "config": config,
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "run_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    # 同时向 metrics.jsonl 写入最终 summary event
    metrics_path = output_dir / "metrics.jsonl"

    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_type": "experiment_summary",
                    **summary,
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

    print(
        json.dumps(
            {"experiment_summary": summary},
            ensure_ascii=False,
            default=str,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pairwise reward-model preference training")
    parser.add_argument("--config", type=Path, default=Path("configs/training/reward_model.json"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args(argv)
    try:
        run_training(args.config, args.model_path, args.resume_from_checkpoint)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
