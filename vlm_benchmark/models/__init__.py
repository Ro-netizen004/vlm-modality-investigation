"""Model registry and unified VLM inference interface."""

from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig

from .base import ModelAdapter, build_prompt
from .internvl import InternVLAdapter
from .llava import LlavaAdapter
from .minicpm import MiniCPMAdapter
from .qwen import QwenAdapter

_ADAPTERS: list[ModelAdapter] = [
    LlavaAdapter(),
    QwenAdapter(),
    MiniCPMAdapter(),
    InternVLAdapter(),
]


def _get_adapter_for_model_id(model_id: str) -> ModelAdapter:
    for adapter in _ADAPTERS:
        if adapter.matches(model_id):
            return adapter
    supported = ", ".join(a.family for a in _ADAPTERS)
    raise ValueError(f"Unsupported VLM model_id={model_id!r}. Supported families: {supported}")


def _get_adapter_by_family(family: str) -> ModelAdapter:
    for adapter in _ADAPTERS:
        if adapter.family == family:
            return adapter
    supported = ", ".join(a.family for a in _ADAPTERS)
    raise ValueError(f"Unknown loaded model family={family!r}. Supported: {supported}")


def load_model(cfg: dict):
    model_id = cfg["model_id"]
    use_4bit = cfg["use_4bit"]
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )
    adapter = _get_adapter_for_model_id(model_id)
    processor, model = adapter.load(model_id=model_id, use_4bit=use_4bit, bnb_config=bnb_config)
    model.eval()
    setattr(model, "_vlm_family", adapter.family)
    print(f"Model loaded ({adapter.family}).")
    return processor, model


def run_inference(processor, model, question: str | None, image, cfg: dict) -> str:
    mode = cfg["experiment_mode"]
    family = getattr(model, "_vlm_family", None)
    if family is None:
        raise ValueError("Loaded model missing _vlm_family; use load_model() from vlm_benchmark.models.")
    adapter = _get_adapter_by_family(family)
    prompt = build_prompt(adapter.family, question, mode)
    formatted = adapter.format_input(processor=processor, model=model, prompt=prompt, image=image, mode=mode)
    return adapter.infer(processor=processor, model=model, formatted=formatted, cfg=cfg)

