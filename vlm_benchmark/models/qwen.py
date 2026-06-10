"""Qwen2-VL model adapter."""

from __future__ import annotations

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration


class QwenAdapter:
    family = "qwen2_vl"

    def matches(self, model_id: str) -> bool:
        m = model_id.lower()
        return "qwen2-vl" in m or ("qwen" in m and "vl" in m)

    def load(self, model_id: str, use_4bit: bool, bnb_config: BitsAndBytesConfig | None):
        print(f"Loading Qwen2-VL processor: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)
        print("Loading Qwen2-VL model ...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if use_4bit else None,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        return processor, model

    def format_input(self, processor, model, prompt: str, image, mode: str) -> dict:
        needs_image = mode in ("image_only", "text_and_image", "mismatch")
        if needs_image and image is not None:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(model.device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat_text], return_tensors="pt").to(model.device)
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

