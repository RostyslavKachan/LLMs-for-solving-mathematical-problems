"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Utilities for extracting and comparing numerical answers
             from model output and GSM8K ground-truth answers.
"""

import re
from fractions import Fraction
from decimal import Decimal, InvalidOperation


def extract_label_from_gsm8k(answer_text: str) -> str:
    m = re.search(r"####\s*(.+)", answer_text)
    if not m:
        return answer_text.strip()
    return m.group(1).strip()


def extract_answer_from_model(output_text: str) -> str:
    text = output_text.strip()
    text = text.replace(",", "")

    m = re.search(r"####\s*([-+]?\d+(?:/\d+|\.\d+)?)", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        content = m.group(1).strip()
        nums = re.findall(r"[-+]?\d+(?:/\d+|\.\d+)?", content)
        return nums[0] if nums else content

    nums = re.findall(r"[-+]?\d+(?:/\d+|\.\d+)?", text)
    return nums[-1] if nums else ""


def parse_number(s: str):
    s = s.strip().replace("−", "-")
    try:
        if "/" in s:
            return Fraction(s)
        else:
            return Fraction(Decimal(s))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def numbers_equal_strict(a, b) -> bool:
    fa = a if isinstance(a, Fraction) else parse_number(str(a))
    fb = b if isinstance(b, Fraction) else parse_number(str(b))
    if fa is None or fb is None:
        return False
    return fa == fb
