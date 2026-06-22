"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Entry point for launching LoRA fine-tuning.
             Parses training arguments and delegates to FinetuneModel.
"""

import argparse
from finetune_model import FinetuneModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["chat", "instruct", "base"],
    )
    parser.add_argument("--output-dir", type=str, default="finetuned_model")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=150)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=20)

    parser.add_argument("--run-name", type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    FinetuneModel(
        train_data=args.train_data,
        val_split=args.val_split,
        model=args.model,
        model_type=args.model_type,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        run_name=args.run_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
