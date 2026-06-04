"""Pluggable answer extraction strategies per dataset type."""

from __future__ import annotations

import re
from typing import Protocol


class AnswerStrategy(Protocol):
    dataset_type: str

    def extract(self, text: str) -> float | None:
        ...

    def match(self, prediction: str, reference: str) -> bool:
        ...


class GSM8KAnswerStrategy:
    """GSM8K canonical #### marker, then last number; integer exact match."""

    dataset_type = "gsm8k"

    def extract(self, text: str) -> float | None:
        if not text or not str(text).strip():
            return None

        m = re.search(r"####\s*([\d,\.]+)", str(text))
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass

        nums = re.findall(r"[\d,]+\.?\d*", str(text))
        valid = [n for n in nums if n.replace(",", "").replace(".", "")]
        if valid:
            try:
                return float(valid[-1].replace(",", ""))
            except ValueError:
                pass
        return None

    def match(self, prediction: str, reference: str) -> bool:
        p, r = self.extract(prediction), self.extract(reference)
        if p is None or r is None:
            return False
        return round(p) == round(r)


class SVAMPAnswerStrategy:
    """SVAMP: reference is usually a plain number; pred uses last number fallback."""

    dataset_type = "svamp"

    def extract(self, text: str) -> float | None:
        if text is None:
            return None
        s = str(text).strip()
        if not s:
            return None
        try:
            return float(s.replace(",", ""))
        except ValueError:
            pass

        nums = re.findall(r"-?[\d,]+\.?\d*", s)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
        return None

    def match(self, prediction: str, reference: str) -> bool:
        p, r = self.extract(prediction), self.extract(reference)
        if p is None or r is None:
            return False
        return abs(p - r) < 1e-6


_STRATEGIES: dict[str, AnswerStrategy] = {
    "gsm8k": GSM8KAnswerStrategy(),
    "svamp": SVAMPAnswerStrategy(),
}


def get_answer_strategy(dataset_type: str) -> AnswerStrategy:
    key = dataset_type.lower().strip()
    if key not in _STRATEGIES:
        known = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"Unknown dataset_type={dataset_type!r} for answer parsing. Known: {known}")
    return _STRATEGIES[key]


def extract_answer(text: str, dataset_type: str) -> float | None:
    return get_answer_strategy(dataset_type).extract(text)


def answers_match(prediction: str, reference: str, dataset_type: str) -> bool:
    return get_answer_strategy(dataset_type).match(prediction, reference)
