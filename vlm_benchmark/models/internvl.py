"""InternVL model adapter."""

from __future__ import annotations

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class InternVLAdapter:
    family = "internvl"

    def matches(self, model_id: str) -> bool:
        return "internvl" in model_id.lower()

    def load(self, model_id: str, use_4bit: bool, bnb_config: BitsAndBytesConfig | None):
        print(f"Loading InternVL tokenizer: {model_id}")
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Loading InternVL model ...")
        if use_4bit:
            print("InternVL: ignoring use_4bit=True (trust_remote_code models may not support it).")
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        return processor, model

    def format_input(self, processor, model, prompt: str, image, mode: str) -> dict:
        needs_image = mode in ("image_only", "text_and_image", "mismatch")
        return {"chat_image": image if needs_image else None, "chat_prompt": prompt}

    def infer(self, processor, model, formatted: dict, cfg: dict) -> str:
        if not hasattr(model, "chat"):
            raise AttributeError("InternVL model does not expose .chat(); check model_id compatibility.")
        chat_image = formatted["chat_image"]
        chat_prompt = formatted["chat_prompt"]
        try:
            response = model.chat(image=chat_image, msgs=[{"role": "user", "content": chat_prompt}], tokenizer=processor)
        except TypeError:
            try:
                response = model.chat(processor, chat_image, chat_prompt)
            except TypeError:
                response = model.chat(tokenizer=processor, pixel_values=chat_image, question=chat_prompt)
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()

