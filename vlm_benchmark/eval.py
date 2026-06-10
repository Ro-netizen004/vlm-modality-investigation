"""Answer extraction and evaluation utilities."""

from __future__ import annotations

import re

from .answer_parsing import answers_match as _answers_match
from .answer_parsing import extract_answer, get_answer_strategy


def extract_numeric_answer(text: str, dataset_type: str = "gsm8k") -> float | None:
    """Backward-compatible alias for extract_answer."""
    return extract_answer(text, dataset_type)


def answers_match(prediction: str, reference: str, dataset_type: str = "gsm8k") -> bool:
    return _answers_match(prediction, reference, dataset_type)


def classify_error(prediction: str, reference: str, dataset_type: str = "gsm8k") -> str:
    if answers_match(prediction, reference, dataset_type):
        return "correct"
    if extract_answer(prediction, dataset_type) is None:
        return "no_number"
    vision_keywords = [
        "cannot read",
        "can't read",
        "image is unclear",
        "unable to see",
        "blurry",
        "illegible",
        "not visible",
        "cannot see the text",
        "cannot identify",
    ]
    if any(kw in prediction.lower() for kw in vision_keywords):
        return "vision_error"
    return "arithmetic_error" if re.search(r"[\+\-\*\/\=]", prediction) else "reasoning_error"


def compute_accuracy(correct_flags: list[bool]) -> float:
    return sum(correct_flags) / len(correct_flags) if correct_flags else 0.0


def score_mismatch(prediction: str, image_ref: str, text_ref: str, dataset_type: str = "gsm8k") -> dict:
    strategy = get_answer_strategy(dataset_type)
    pred_val = strategy.extract(prediction)
    img_val = strategy.extract(image_ref)
    txt_val = strategy.extract(text_ref)

    if pred_val is None or img_val is None or txt_val is None:
        return {"follows": "invalid", "img_diff": None, "txt_diff": None}

    img_diff = abs(pred_val - img_val)
    txt_diff = abs(pred_val - txt_val)
    if img_diff < txt_diff:
        follows = "image"
    elif txt_diff < img_diff:
        follows = "text"
    else:
        follows = "equal"

    return {
        "follows": follows,
        "img_diff": round(img_diff, 4),
        "txt_diff": round(txt_diff, 4),
    }
