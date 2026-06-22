"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Prepares the GSM8K dataset for fine-tuning by formatting
             examples into chat, instruct, or base format and saving
             the result as a JSONL file.
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from prompts import SYSTEM_INSTRUCTION


def format_example(question: str, answer: str, model_type: str) -> dict:
    if model_type == "chat":
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }

    if model_type == "instruct":
        return {
            "messages": [
                {
                    "role": "user",
                    "content": SYSTEM_INSTRUCTION + "\n\n" + question,
                },
                {"role": "assistant", "content": answer},
            ]
        }

    return {"text": f"Question: {question}\nAnswer: {answer}"}


def prepare_dataset(split: str, model_type: str, output_dir: str):
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    prepared = []

    for example in tqdm(dataset, desc=f"Preparing {split}"):
        prepared.append(
            format_example(example["question"], example["answer"], model_type)
        )

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    jsonl_file = output_path / f"gsm8k_{split}_prepared.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for ex in prepared:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["chat", "instruct", "base"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prepared_data",
    )
    args = parser.parse_args()
    prepare_dataset(
        split=args.split, model_type=args.model_type, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
