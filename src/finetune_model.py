"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Core fine-tuning pipeline: loads data, applies LoRA adapters,
             configures the HuggingFace Trainer, and saves the final
             LoRA adapter.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed as hf_set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from tokenize_utils import tokenize_chat, tokenize_base

from callbacks import RealAccuracyCallback, TrainerCallback


def load_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_data(
    train_data: list[dict], val_split: float, output_dir: str
) -> tuple[list[dict], list[dict], Path | None]:
    val_size = int(len(train_data) * val_split)
    train_data_split = train_data[:-val_size] if val_size > 0 else train_data
    val_data = train_data[-val_size:] if val_size > 0 else []

    val_jsonl_path = None
    if val_data:
        val_jsonl_path = Path(output_dir) / "gsm8k_val_prepared.jsonl"
        val_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(val_jsonl_path, "w", encoding="utf-8") as f:
            for example in val_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return train_data_split, val_data, val_jsonl_path


def add_callbacks(
    base_model: Any,
    tokenizer: AutoTokenizer,
    val_jsonl_path: Path | None,
    output_dir: str,
    model_type: str,
    val_data: list[dict],
    val_dataset: Dataset | None,
) -> list[TrainerCallback]:
    callbacks: list[TrainerCallback] = []

    if val_dataset and val_jsonl_path:
        accuracy_callback = RealAccuracyCallback(
            model=base_model,
            tokenizer=tokenizer,
            val_jsonl_path=str(val_jsonl_path),
            output_dir=output_dir,
            model_type=model_type,
            num_samples=len(val_data),
            batch_size=16,
        )
        callbacks.append(accuracy_callback)

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=6, early_stopping_threshold=0.001
        )
        callbacks.append(early_stopping)

    return callbacks


class FinetuneModel:

    def __init__(
        self,
        train_data: str,
        val_split: float,
        model: str,
        model_type: str,
        output_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        gradient_accumulation_steps: int,
        warmup_steps: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        eval_steps: int | None,
        save_steps: int | None,
        run_name: str | None,
        seed: int,
    ):
        set_seed(seed)

        data = load_jsonl(train_data)

        train_data_split, val_data, val_jsonl_path = prepare_data(
            data, val_split, output_dir
        )

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        base_model: Any = get_peft_model(base_model, lora_config)
        base_model.enable_input_require_grads()
        base_model.config.use_cache = False
        base_model.print_trainable_parameters()

        is_base = model_type == "base"

        tokenize_fn = (
            (lambda x: tokenize_base(x, tokenizer))
            if is_base
            else (lambda x: tokenize_chat(x, tokenizer))
        )

        train_dataset = Dataset.from_list(train_data_split)
        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train",
        )

        val_dataset: Dataset | None = None
        if val_data:
            val_dataset = Dataset.from_list(val_data)
            val_dataset = val_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=val_dataset.column_names,
                desc="Tokenizing validation",
            )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=base_model, padding=True
        )

        default_run = f"{model_type}-gsm8k-lr{learning_rate}-r{lora_r}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=5,
            save_steps=save_steps if save_steps else None,
            eval_steps=eval_steps if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",
            save_total_limit=10,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="val_accuracy" if val_dataset else None,
            greater_is_better=True if val_dataset else None,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="wandb",
            run_name=run_name if run_name else default_run,
            remove_unused_columns=False,
            seed=seed,
            data_seed=seed,
        )

        callbacks = add_callbacks(
            base_model,
            tokenizer,
            val_jsonl_path,
            output_dir,
            model_type,
            val_data,
            val_dataset,
        )

        trainer = Trainer(
            model=base_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
        )

        trainer.train()

        final_output_dir = Path(output_dir) / "final"
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
