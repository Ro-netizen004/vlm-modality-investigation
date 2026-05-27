"""Shared contracts and prompt/input helpers for VLM adapters."""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image
from transformers import BitsAndBytesConfig


class ModelAdapter(Protocol):
    family: str

    def matches(self, model_id: str) -> bool: ...

    def load(self, model_id: str, use_4bit: bool, bnb_config: BitsAndBytesConfig | None): ...

    def format_input(
        self,
        processor: Any,
        model: Any,
        prompt: str,
        image: Image.Image | None,
        mode: str,
    ) -> dict: ...

    def infer(self, processor: Any, model: Any, formatted: dict, cfg: dict) -> str: ...


def build_prompt(model_family: str, question: str | None, mode: str) -> str:
    instruction = (
        "Solve the following math problem step by step. "
        "Show your reasoning, then end with '#### <number>'."
    )
    img_token = "<image>\n" if model_family == "llava" else ""

    if mode == "text_only":
        return f"[INST] {instruction}\n\nProblem: {question} [/INST]"

    if mode == "image_only":
        return (
            f"[INST] {img_token}"
            "The image shows a math word problem. Read it carefully and solve it "
            "step by step. End with '#### <number>'. [/INST]"
        )

    if mode in ("text_and_image", "mismatch"):
        return f"[INST] {img_token}{instruction}\n\nProblem: {question} [/INST]"

    raise ValueError(f"Unknown mode: {mode}")

