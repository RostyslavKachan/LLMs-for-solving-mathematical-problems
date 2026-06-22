"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Custom HuggingFace Trainer callback that computes real
             generation accuracy on the validation set at each eval
             step and logs results to Weights & Biases and a README.
"""

import json
import traceback
from pathlib import Path
import torch
import wandb

from transformers import TrainerCallback, StoppingCriteriaList

from answer_parser import extract_answer_from_model, parse_number, numbers_equal_strict
from first_answer_stopping import FirstAnswerStopping
from prompts import FEW_SHOT_EXAMPLES
from update_readme import update_readme, init_readme


class RealAccuracyCallback(TrainerCallback):

    def __init__(
        self,
        model,
        tokenizer,
        val_jsonl_path: str,
        output_dir: str,
        model_type: str = "chat",
        num_samples: int = 256,
        batch_size: int = 16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_jsonl_path = val_jsonl_path
        self.output_dir = output_dir
        self.model_type = model_type
        self.num_samples = num_samples
        self.batch_size = batch_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eval_data = self._load_eval_data()

        self.readme_path = Path(self.output_dir) / "README.md"
        init_readme(self.readme_path, self.output_dir)

    def _load_eval_data(self) -> list[dict]:
        eval_data = []

        with open(self.val_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)

                if self.model_type == "base":
                    question = example.get("text", "")
                    if "Question:" in question and "Answer:" in question:
                        q_part = question.split("Answer:")[0]
                        q_part = q_part.replace("Question:", "").strip()
                    else:
                        q_part = question
                    a_part = (
                        example.get("text", "").split("Answer:")[-1].strip()
                        if "Answer:" in example.get("text", "")
                        else ""
                    )
                    ground_truth = extract_answer_from_model(a_part)
                    eval_data.append(
                        {
                            "question": q_part,
                            "ground_truth": ground_truth,
                        }
                    )
                else:
                    messages = example["messages"]
                    prompt_messages = [m for m in messages if m["role"] != "assistant"]
                    assistant_msg = next(
                        (m for m in messages if m["role"] == "assistant"), None
                    )
                    ground_truth = (
                        extract_answer_from_model(assistant_msg["content"])
                        if assistant_msg
                        else ""
                    )
                    eval_data.append(
                        {
                            "prompt_messages": prompt_messages,
                            "ground_truth": ground_truth,
                        }
                    )

        if len(eval_data) > self.num_samples:
            eval_data = eval_data[: self.num_samples]

        return eval_data

    def _build_prompt(self, item: dict) -> str:
        if self.model_type == "base":
            return f"{FEW_SHOT_EXAMPLES}Question: {item['question']}\nAnswer:"

        return self.tokenizer.apply_chat_template(
            item["prompt_messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prepare_batch(
        self, batch_data: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        prompts = []
        ground_truths = []

        for item in batch_data:
            prompts.append(self._build_prompt(item))
            ground_truths.append(item["ground_truth"])

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        )

        return inputs["input_ids"], inputs["attention_mask"], ground_truths
    # same as in main.py but generate for batch
    def _generate_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        rep_penalty = 1.1 if self.model_type == "base" else 1.0
        stopping_criteria = None
        if self.model_type == "base":
            stopping_criteria = StoppingCriteriaList(
                [FirstAnswerStopping(self.tokenizer, input_ids.shape[1])]
            )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                repetition_penalty=rep_penalty,
                stopping_criteria=stopping_criteria,
            )
        return outputs

    def _extract_and_evaluate(
        self,
        outputs: torch.Tensor,
        prompt_lengths: list[int],
        ground_truths: list[str],
    ) -> tuple[int, int]:
        correct = 0
        total = len(ground_truths)

        for i, (output, prompt_len, gold) in enumerate(
            zip(outputs, prompt_lengths, ground_truths)
        ):
            try:
                continuation_ids = output[prompt_len:]
                generated_text = self.tokenizer.decode(
                    continuation_ids, skip_special_tokens=True
                )
                pred_number = extract_answer_from_model(generated_text)
                pred_parsed = parse_number(pred_number)
                gold_parsed = parse_number(gold)

                if numbers_equal_strict(pred_parsed, gold_parsed):
                    correct += 1

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue

        return correct, total

    def _evaluate_accuracy(self) -> tuple[float, int, int]:

        self.model.eval()

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        total_correct = 0
        total_examples = 0

        for i in range(0, len(self.eval_data), self.batch_size):
            batch_data = self.eval_data[i : i + self.batch_size]

            input_ids, attention_mask, ground_truths = self._prepare_batch(batch_data)
            prompt_lengths = [input_ids.shape[1]] * len(ground_truths)
            outputs = self._generate_batch(input_ids, attention_mask)
            correct, total = self._extract_and_evaluate(
                outputs, prompt_lengths, ground_truths
            )

            total_correct += correct
            total_examples += total

        self.tokenizer.padding_side = original_padding_side

        accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        return accuracy, total_correct, total_examples

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):

        try:
            accuracy, correct, total = self._evaluate_accuracy()

            if metrics is not None:
                metrics["eval_val_accuracy"] = accuracy

            if wandb.run is not None:
                wandb.log(
                    {
                        "val_accuracy": accuracy,
                        "step": state.global_step,
                        "epoch": state.epoch,
                    }
                )

            update_readme(
                self.readme_path,
                state.global_step,
                accuracy,
                correct,
                total,
                state.epoch,
            )

        except Exception as e:
            traceback.print_exc()

            if metrics is not None:
                metrics["eval_val_accuracy"] = 0.0
