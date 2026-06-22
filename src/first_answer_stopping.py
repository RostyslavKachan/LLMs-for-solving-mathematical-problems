"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Custom stopping criteria that halts text generation as soon
             as the pattern '#### <number>' appears in the model output.
"""

import re
from transformers import StoppingCriteria


class FirstAnswerStopping(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.triggered = None

    def __call__(self, input_ids, scores, **kwargs):
        batch_size = input_ids.shape[0]
        if self.triggered is None:
            self.triggered = [False] * batch_size

        for i in range(batch_size):
            if not self.triggered[i]:
                new_ids = input_ids[i][self.prompt_len :]
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                if re.search(r"####\s*[-+]?\d+(?:/\d+|\.\d+)?(?=\s)", text):
                    self.triggered[i] = True

        return all(self.triggered)
