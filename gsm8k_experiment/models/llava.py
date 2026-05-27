"""LLaVA model adapter."""

from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor


class LlavaAdapter:
    family = "llava"

    def matches(self, model_id: str) -> bool:
        return "llava" in model_id.lower()

    def load(self, model_id: str, use_4bit: bool, bnb_config: BitsAndBytesConfig | None):
        print(f"Loading LLaVA-NeXT processor: {model_id}")
        processor = LlavaNextProcessor.from_pretrained(model_id)
        print("Loading LLaVA-NeXT model ...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if use_4bit else None,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        return processor, model

    def format_input(self, processor, model, prompt: str, image, mode: str) -> dict:
        needs_image = mode in ("image_only", "text_and_image", "mismatch")
        if needs_image and image is not None:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        return {"inputs": inputs}

    def infer(self, processor, model, formatted: dict, cfg: dict) -> str:
        inputs = formatted["inputs"]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=cfg["do_sample"],
            )
        n_prompt_tokens = inputs["input_ids"].shape[1]
        return processor.decode(out[0][n_prompt_tokens:], skip_special_tokens=True).strip()

