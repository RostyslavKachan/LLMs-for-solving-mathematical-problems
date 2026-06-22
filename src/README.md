# Large language models for solving mathematical problems

**Author:** Rostyslav Kachan (xkacha02)
**Year:** 2026
**Institution:** Brno University of Technology, Faculty of Information Technology

## Overview

This project fine-tunes three open-source 7B language models on the GSM8K dataset
using LoRA (Low-Rank Adaptation) to improve mathematical reasoning capabilities.
The models are evaluated using exact-match accuracy and process reward model (PRM) scoring
via Qwen2.5-Math-PRM-7B.

**Models:**

- `meta-llama/Llama-2-7b-hf` — base model
- `meta-llama/Llama-2-7b-chat-hf` — chat model (SFT + RLHF)
- `mistralai/Mistral-7B-Instruct-v0.1` — instruct model (SFT)

## Project Structure

```
src/
├── prepare_dataset.py          # Prepares GSM8K data in chat/instruct/base format
├── finetune.py                 # Entry point for LoRA fine-tuning
├── finetune_model.py           # Core fine-tuning pipeline
├── tokenize_utils.py           # Tokenization with label masking
├── callbacks.py                # Real-time accuracy evaluation during training
├── first_answer_stopping.py    # Stopping criteria for base model generation
├── prompts.py                  # Prompt templates and few-shot examples
├── main.py                     # Evaluation script (exact-match accuracy)
├── evaluate_prm.py             # PRM scoring with Qwen2.5-Math-PRM-7B
├── answer_parser.py            # Answer extraction and comparison utilities
├── update_readme.py            # Create and update README for fine-tuning logs
└── requirements.txt            # Dependencies

```

## Requirements

- Python 3.11
- CUDA 12.8
- GPU

### Dependencies


| Library      | Version           |
| ------------ | ----------------- |
| torch        | 2.9.1 (CUDA 12.8) |
| transformers | 4.57.3            |
| peft         | 0.18.0            |
| datasets     | 4.4.1             |
| wandb        | 0.23.1            |
| numpy        | 2.3.5             |
| tqdm         | 4.67.1            |


## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

```bash
python3 prepare_dataset.py \
    --split train \
    --model-type chat \
    --output-dir ./data
```

`--model-type` options: `chat`, `instruct`, `base`
Output: `./data/gsm8k_train_prepared.jsonl`

### 2. Fine-tune

```bash
python3 finetune.py \
    --train-data ./data/gsm8k_train_prepared.jsonl \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type chat \
    --output-dir ./finetuned_model \
    --epochs 15 \
    --batch-size 4 \
    --learning-rate 3e-4 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --gradient-accumulation-steps 32 \
    --val-split 0.15 \
    --eval-steps 20 \
    --save-steps 20 \
    --seed 42
```

The final LoRA adapter is saved to `./finetuned_model/final/`.

### 3. Evaluate (Exact-Match Accuracy)

```bash
# Baseline (no adapter)
python3 main.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type chat \
    --out results_baseline.json

# Fine-tuned
python3 main.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type chat \
    --adapter ./finetuned_model/final \
    --out results_finetuned.json
```

Optional: `--subset N` to evaluate only the first N examples, `--split train|test`.

### 4. PRM Evaluation

```bash
python3 evaluate_prm.py \
    --input results_finetuned.json \
    --output results_prm.json
```

Requires `Qwen/Qwen2.5-Math-PRM-7B` (downloaded automatically from Hugging Face).

## Weights & Biases Setup

Training metrics are logged to [Weights & Biases](https://wandb.ai).
Before fine-tuning, set the following environment variables:

```bash
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="your_project_name"
export WANDB_MODE=online
```

Alternatively, run `wandb login` and follow the instructions.

## Notes

- Model weights are downloaded automatically from Hugging Face on first run.
  Set `HF_HOME` to control the cache location.
- Access to `meta-llama` models requires accepting the license on Hugging Face
  and setting `HF_TOKEN`:
  ```bash
  export HF_TOKEN="your_huggingface_token"
  ```
- The validation split (15% by default) is taken from the end of the training file.

