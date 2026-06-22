"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Evaluation script that loads a model with an optional LoRA
             adapter, generates answers on the GSM8K test set, and
             computes exact-match accuracy.
"""

import argparse
import json
import torch


from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from peft import PeftModel
from prompts import build_chat_prompt, build_instruct_prompt, build_base_prompt
from answer_parser import (
    extract_answer_from_model,
    parse_number,
    numbers_equal_strict,
    extract_label_from_gsm8k,
)
from first_answer_stopping import FirstAnswerStopping


def evaluate_model(
    model_name: str,
    model_type: str,
    split: str = "test",
    subset_size: int | None = None,
    output_json: str | None = None,
    adapter_path: str | None = None,
):
    assert model_type in (
        "chat",
        "instruct",
        "base",
    ), f"--model-type must be one of: chat, instruct, base. Got: {model_type}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model_name = f"{model_name} + {adapter_path}"

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if subset_size is not None and subset_size > 0:
        ds = ds.select(range(min(subset_size, len(ds))))

    model.eval()
    results = []
    correct = 0

    for idx, ex in enumerate(ds):
        q = ex["question"]
        gold_full = ex["answer"]
        gold_label_str = extract_label_from_gsm8k(gold_full)
        gold_num = parse_number(gold_label_str)

        if model_type == "chat":
            prompt_text = build_chat_prompt(q, tokenizer)
        elif model_type == "instruct":
            prompt_text = build_instruct_prompt(q, tokenizer)
        else:
            prompt_text = build_base_prompt(q)

        inputs = tokenizer([prompt_text], return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        rep_penalty = 1.1 if model_type == "base" else 1.0
        stopping_criteria = None
        if model_type == "base":
            stopping_criteria = StoppingCriteriaList(
                [FirstAnswerStopping(tokenizer, inputs["input_ids"].shape[1])]
            )
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=rep_penalty,
                stopping_criteria=stopping_criteria,
            )

        gen_ids = generated_ids[0][inputs["input_ids"].shape[1] :]
        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        model_ans_str = extract_answer_from_model(output_text)
        model_num = parse_number(model_ans_str)

        is_correct = numbers_equal_strict(gold_num, model_num)
        if is_correct:
            correct += 1

        results.append(
            {
                "idx": idx,
                "question": q,
                "gold_full_answer": gold_full,
                "gold_label_str": gold_label_str,
                "gold_label_parsed": str(gold_num) if gold_num is not None else "",
                "model_output_full": output_text,
                "model_answer_str": model_ans_str,
                "model_answer_parsed": str(model_num) if model_num is not None else "",
                "is_correct": int(is_correct),
            }
        )

    total = len(results)
    acc = correct / total if total > 0 else 0.0

    if output_json:
        output_data = {
            "metadata": {
                "model": model_name,
                "model_type": model_type,
                "dataset": "GSM8K",
                "split": split,
                "total_examples": total,
                "correct_answers": correct,
                "accuracy": round(acc, 4),
                "accuracy_percent": f"{acc * 100:.2f}%",
            },
            "results": results,
        }
        output_path = Path(output_json)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["chat", "instruct", "base"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        model_type=args.model_type,
        split=args.split,
        subset_size=args.subset,
        output_json=args.out,
        adapter_path=args.adapter,
    )


if __name__ == "__main__":
    main()
