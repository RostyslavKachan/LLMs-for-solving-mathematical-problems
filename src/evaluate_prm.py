"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Scores each reasoning step in model outputs using the
             Qwen2.5-Math-PRM-7B process reward model and computes
             per-example scores (avg, min, product) and global
             statistics across all evaluated examples.
"""

import json
import argparse
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


# Taken from the official Qwen2.5-Math-PRM-7B model card on Hugging Face.
def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def split_into_steps(text: str) -> list[str]:
    major_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    result = []
    for block in major_blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        result.extend(lines)
    return result if result else ([text.strip()] if text.strip() else [])


def evaluate_with_prm(
    input_json: str,
    output_json: str,
    prm_model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
    subset_size: int | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        prm_model_name,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    ).eval()

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    assert (
        step_sep_id != tokenizer.unk_token_id
    ), "Tokenizer does not recognize <extra_0> token"

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    if subset_size is not None and subset_size > 0:
        results = results[:subset_size]

    all_step_scores = []
    all_avg_scores = []
    all_min_scores = []
    all_product_scores = []

    for idx, example in enumerate(results):
        question = example["question"]
        model_output = example["model_output_full"]

        steps = split_into_steps(model_output)

        if not steps:
            example["prm_step_scores"] = []
            example["prm_avg_score"] = 0.0
            example["prm_min_score"] = 0.0
            example["prm_product_score"] = 0.0
            example["prm_num_steps"] = 0
            all_step_scores.append([])
            all_avg_scores.append(0.0)
            all_min_scores.append(0.0)
            all_product_scores.append(0.0)
            continue

        response_with_sep = "<extra_0>".join(steps) + "<extra_0>"

        messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_with_sep},
        ]

        conversation_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        input_ids = tokenizer.encode(
            conversation_str,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=False)

        token_masks = input_ids == step_sep_id
        step_rewards = make_step_rewards(outputs[0], token_masks)

        scores = step_rewards[0] if step_rewards else []

        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        # Log space avoids floating-point underflow when multiplying many small scores.
        product_score = (
            math.exp(sum(math.log(s + 1e-10) for s in scores)) if scores else 0.0
        )

        example["prm_step_scores"] = [round(s, 4) for s in scores]
        example["prm_avg_score"] = round(avg_score, 4)
        example["prm_min_score"] = round(min_score, 4)
        example["prm_product_score"] = round(product_score, 4)
        example["prm_num_steps"] = len(scores)

        all_step_scores.append(scores)
        all_avg_scores.append(avg_score)
        all_min_scores.append(min_score)
        all_product_scores.append(product_score)

    total = len(results)
    correct_examples = [i for i, ex in enumerate(results) if ex["is_correct"] == 1]
    incorrect_examples = [i for i, ex in enumerate(results) if ex["is_correct"] == 0]

    global_avg = sum(all_avg_scores) / total if total > 0 else 0.0
    global_min_avg = sum(all_min_scores) / total if total > 0 else 0.0

    avg_correct = (
        sum(all_avg_scores[i] for i in correct_examples) / len(correct_examples)
        if correct_examples
        else 0.0
    )
    avg_incorrect = (
        sum(all_avg_scores[i] for i in incorrect_examples) / len(incorrect_examples)
        if incorrect_examples
        else 0.0
    )

    product_correct = (
        sum(all_product_scores[i] for i in correct_examples) / len(correct_examples)
        if correct_examples
        else 0.0
    )
    product_incorrect = (
        sum(all_product_scores[i] for i in incorrect_examples) / len(incorrect_examples)
        if incorrect_examples
        else 0.0
    )

    all_individual_scores = [s for scores in all_step_scores for s in scores]
    high_confidence_steps = sum(1 for s in all_individual_scores if s > 0.8)
    total_steps = len(all_individual_scores)
    high_confidence_rate = (
        high_confidence_steps / total_steps if total_steps > 0 else 0.0
    )

    global_product_avg = sum(all_product_scores) / total if total > 0 else 0.0

    prm_metadata = {
        "prm_model": prm_model_name,
        "total_examples_evaluated": total,
        "global_avg_step_score": round(global_avg, 4),
        "global_avg_min_score": round(global_min_avg, 4),
        "global_avg_product_score": round(global_product_avg, 4),
        "avg_score_correct_answers": round(avg_correct, 4),
        "avg_score_incorrect_answers": round(avg_incorrect, 4),
        "product_score_correct_answers": round(product_correct, 4),
        "product_score_incorrect_answers": round(product_incorrect, 4),
        "total_steps_evaluated": total_steps,
        "high_confidence_steps_rate": round(high_confidence_rate, 4),
    }

    data["metadata"]["prm_evaluation"] = prm_metadata
    data["results"] = results

    output_path = Path(output_json)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"PRM EVALUATION SUMMARY")
    print(f"Examples evaluated:         {total}")
    print(f"Total steps evaluated:      {total_steps}")
    print(f"Global avg step score:      {global_avg:.4f}")
    print(f"Global avg min score:       {global_min_avg:.4f}")
    print(f"Global avg product score:   {global_product_avg:.4f}")
    print(f"Avg score (correct ans):    {avg_correct:.4f}")
    print(f"Avg score (incorrect ans):  {avg_incorrect:.4f}")
    print(f"Product (correct ans):      {product_correct:.4f}")
    print(f"Product (incorrect ans):    {product_incorrect:.4f}")
    print(f"Steps with score > 0.8:     {high_confidence_rate*100:.1f}%")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSON from main.py evaluation"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON with PRM scores"
    )
    parser.add_argument("--prm-model", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument(
        "--subset", type=int, default=None, help="Evaluate only first N examples"
    )
    args = parser.parse_args()

    evaluate_with_prm(
        input_json=args.input,
        output_json=args.output,
        prm_model_name=args.prm_model,
        subset_size=args.subset,
    )


if __name__ == "__main__":
    main()
