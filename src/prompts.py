"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Prompt templates and few-shot examples for chat, instruct,
             and base model types used during evaluation and fine-tuning.
"""

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. "
    "Solve the following problem step by step and provide your final answer clearly."
)

FEW_SHOT_EXAMPLES = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100/2 = $50. Betty's grandparents gave her 15 * 2 = $30. This means, Betty needs 100 - 50 - 30 - 15 = $5 more. #### 5

"""


def build_chat_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_instruct_prompt(question: str, tokenizer) -> str:
    messages = [
        {
            "role": "user",
            "content": SYSTEM_INSTRUCTION + "\n\n" + question,
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_base_prompt(question: str) -> str:
    return f"{FEW_SHOT_EXAMPLES}Question: {question}\nAnswer:"
