"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Tokenization utilities for chat/instruct and base model
             formats, including label masking so the loss is computed
             only on the assistant/answer tokens.
"""


def tokenize_chat(examples, tokenizer):
    texts = []
    prompt_texts = []

    for messages in examples["messages"]:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(full_text)

        prompt_only = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_texts.append(prompt_only)

    model_inputs = tokenizer(texts, truncation=True, max_length=512, padding=False)

    prompt_inputs = tokenizer(
        prompt_texts, truncation=True, max_length=512, padding=False
    )

    labels = []
    for i, input_ids in enumerate(model_inputs["input_ids"]):
        label = input_ids.copy()
        prompt_len = len(prompt_inputs["input_ids"][i])
        for j in range(min(prompt_len, len(label))):
            label[j] = -100
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs


def tokenize_base(examples, tokenizer):
    texts = examples["text"]

    model_inputs = tokenizer(texts, truncation=True, max_length=512, padding=False)
    labels = []
    answer_token_ids = tokenizer.encode("\nAnswer:", add_special_tokens=False)

    for input_ids in model_inputs["input_ids"]:
        label = input_ids.copy()

        answer_start = -1
        for k in range(len(input_ids) - len(answer_token_ids) + 1):
            if input_ids[k : k + len(answer_token_ids)] == answer_token_ids:
                answer_start = k + len(answer_token_ids)

        if answer_start > 0:
            for j in range(min(answer_start, len(label))):
                label[j] = -100
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs
