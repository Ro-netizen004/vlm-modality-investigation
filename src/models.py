"""
Model loading and inference for supported VLM architectures.

All models are 100% free, open-source, and run on Google Colab T4 (16GB VRAM)
or Kaggle P100/T4 with 4-bit quantization via bitsandbytes.

Supported models:
  - Qwen2-VL-2B-Instruct          (2B,  qwen)
  - Qwen2.5-VL-7B-Instruct        (7B,  qwen)
  - LLaVA-v1.6-Mistral-7B         (7B,  llava)
  - LLaVA-OneVision-Qwen2-7B      (7B,  llava_onevision)
  - InternVL2-8B                   (8B,  internvl)
  - Phi-3.5-Vision-Instruct        (4B,  phi)
  - MiniCPM-V-2.6                  (8B,  minicpm)
  - Idefics3-8B-Llama3             (8B,  idefics)
"""

import gc
import torch
from PIL import Image


# ── Shared quantization config ────────────────────────────────────────────────

def get_quant_config(quantize, compute_dtype):
    """Get BitsAndBytesConfig for 4-bit or 8-bit quantization."""
    if quantize is None:
        return None
    from transformers import BitsAndBytesConfig
    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,  # saves extra memory
        )
    elif quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ── Standard math prompts ─────────────────────────────────────────────────────

TEXT_ONLY_PROMPT = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then end with '#### <answer>'.\n\n"
    "Problem: {question}"
)

IMAGE_PROMPT = (
    "The image contains a math word problem. "
    "Read it carefully and solve it step by step. "
    "End with '#### <answer>'."
)


class VLMModel:
    """Unified interface for VLM inference across architectures."""

    def __init__(self, model_name, model_type, max_new_tokens=256,
                 torch_dtype="bfloat16", quantize=None, attn_implementation=None):
        self.model_name = model_name
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.dtype = getattr(torch, torch_dtype)
        self.quantize = quantize
        # Set to "eager" when attention weights are needed (output_attentions=True);
        # the default SDPA backend returns None for attentions.
        self.attn_implementation = attn_implementation
        self.model = None
        self.processor = None
        self.tokenizer = None  # some models need a separate tokenizer

    def load(self):
        """Load model and processor."""
        print(f"Loading {self.model_name} (type={self.model_type}, quant={self.quantize})...")

        loader = {
            "qwen": self._load_qwen,
            "llava": self._load_llava,
            "llava_onevision": self._load_llava_onevision,
            "internvl": self._load_internvl,
            "phi": self._load_phi,
            "minicpm": self._load_minicpm,
            "idefics": self._load_idefics,
        }

        if self.model_type not in loader:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Supported: {list(loader.keys())}"
            )

        loader[self.model_type]()
        print(f"Model loaded: {self.model_name}")
        self._print_memory_usage()

    def _print_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL LOADERS
    # ══════════════════════════════════════════════════════════════════════════

    def _load_qwen(self):
        """Qwen2-VL and Qwen2.5-VL models."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False,
        )
        quant_config = get_quant_config(self.quantize, self.dtype)
        kwargs = {"device_map": "auto", "torch_dtype": self.dtype, "trust_remote_code": True}
        if quant_config:
            kwargs["quantization_config"] = quant_config
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

    def _load_llava(self):
        """LLaVA-v1.6 (LLaVA-NeXT) with Mistral backbone."""
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        quant_config = get_quant_config(self.quantize, self.dtype)
        kwargs = {"device_map": "auto", "torch_dtype": self.dtype}
        if quant_config:
            kwargs["quantization_config"] = quant_config
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs)
        self.model.eval()

    def _load_llava_onevision(self):
        """LLaVA-OneVision (latest LLaVA with Qwen2 backbone)."""
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True)
        quant_config = get_quant_config(self.quantize, self.dtype)
        kwargs = {"device_map": "auto", "torch_dtype": self.dtype, "trust_remote_code": True}
        if quant_config:
            kwargs["quantization_config"] = quant_config
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs)
        self.model.eval()

    def _load_internvl(self):
        """InternVL2 models — use AutoModel with trust_remote_code."""
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        quant_config = get_quant_config(self.quantize, self.dtype)
        kwargs = {"device_map": "auto", "torch_dtype": self.dtype, "trust_remote_code": True}
        if quant_config:
            kwargs["quantization_config"] = quant_config
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

    def _load_phi(self):
        """Phi-3.5-Vision-Instruct from Microsoft."""
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True)
        quant_config = get_quant_config(self.quantize, self.dtype)
        kwargs = {"device_map": "auto", "torch_dtype": self.dtype,
                  "trust_remote_code": True, "_attn_implementation": "eager"}
        if quant_config:
            kwargs["quantization_config"] = quant_config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

    def _load_minicpm(self):
        """MiniCPM-V-2.6 from OpenBMB."""
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        quant_config = get_quant_config(self.quantize, self.dtype)
        kwargs = {"device_map": "auto", "torch_dtype": self.dtype, "trust_remote_code": True}
        if quant_config:
            kwargs["quantization_config"] = quant_config
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

    def _load_idefics(self):
        """Idefics3-8B-Llama3 from HuggingFace."""
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        quant_config = get_quant_config(self.quantize, self.dtype)
        kwargs = {"device_map": "auto", "torch_dtype": self.dtype}
        if quant_config:
            kwargs["quantization_config"] = quant_config
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs)
        self.model.eval()

    # ══════════════════════════════════════════════════════════════════════════
    #  INFERENCE DISPATCH
    # ══════════════════════════════════════════════════════════════════════════

    def generate_text_only(self, question: str) -> str:
        """Condition 1: text-only, vision encoder unused."""
        dispatch = {
            "qwen": self._qwen_text_only,
            "llava": self._llava_text_only,
            "llava_onevision": self._llava_onevision_text_only,
            "internvl": self._internvl_text_only,
            "phi": self._phi_text_only,
            "minicpm": self._minicpm_text_only,
            "idefics": self._idefics_text_only,
        }
        return dispatch[self.model_type](question)

    def generate_with_image(self, image: Image.Image, text_prompt: str = None) -> str:
        """Condition 2/3: image-based inference, vision encoder active."""
        dispatch = {
            "qwen": self._qwen_with_image,
            "llava": self._llava_with_image,
            "llava_onevision": self._llava_onevision_with_image,
            "internvl": self._internvl_with_image,
            "phi": self._phi_with_image,
            "minicpm": self._minicpm_with_image,
            "idefics": self._idefics_with_image,
        }
        return dispatch[self.model_type](image, text_prompt)

    # ══════════════════════════════════════════════════════════════════════════
    #  QWEN (Qwen2-VL, Qwen2.5-VL)
    # ══════════════════════════════════════════════════════════════════════════

    def _qwen_text_only(self, question):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": TEXT_ONLY_PROMPT.format(question=question)}
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], return_tensors="pt")
        return self._generate(inputs)

    def _qwen_with_image(self, image, text_prompt=None):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text_prompt or IMAGE_PROMPT},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        return self._generate(inputs)

    # ══════════════════════════════════════════════════════════════════════════
    #  LLAVA-1.6 (LLaVA-NeXT Mistral)
    # ══════════════════════════════════════════════════════════════════════════

    def _llava_text_only(self, question):
        prompt = f"[INST] {TEXT_ONLY_PROMPT.format(question=question)} [/INST]"
        inputs = self.processor(text=prompt, return_tensors="pt")
        return self._generate(inputs)

    def _llava_with_image(self, image, text_prompt=None):
        prompt = f"[INST] <image>\n{text_prompt or IMAGE_PROMPT} [/INST]"
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        return self._generate(inputs)

    # ══════════════════════════════════════════════════════════════════════════
    #  LLAVA-ONEVISION (Qwen2 backbone)
    # ══════════════════════════════════════════════════════════════════════════

    def _llava_onevision_text_only(self, question):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": TEXT_ONLY_PROMPT.format(question=question)}
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt")
        return self._generate(inputs)

    def _llava_onevision_with_image(self, image, text_prompt=None):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text_prompt or IMAGE_PROMPT},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        return self._generate(inputs)

    # ══════════════════════════════════════════════════════════════════════════
    #  INTERNVL2 (custom chat API)
    # ══════════════════════════════════════════════════════════════════════════

    def _internvl_text_only(self, question):
        prompt = TEXT_ONLY_PROMPT.format(question=question)
        # InternVL2 uses model.chat() API
        response = self.model.chat(
            self.tokenizer, None, prompt,
            generation_config={"max_new_tokens": self.max_new_tokens, "do_sample": False},
        )
        return response.strip()

    def _internvl_with_image(self, image, text_prompt=None):
        prompt = text_prompt or IMAGE_PROMPT
        pixel_values = self._internvl_process_image(image)
        response = self.model.chat(
            self.tokenizer, pixel_values, prompt,
            generation_config={"max_new_tokens": self.max_new_tokens, "do_sample": False},
        )
        return response.strip()

    def _internvl_process_image(self, image):
        """Process image for InternVL2 using its dynamic preprocessing."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        pixel_values = transform(image).unsqueeze(0).to(self.dtype)
        return pixel_values.to(self.model.device)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHI-3.5-VISION
    # ══════════════════════════════════════════════════════════════════════════

    def _phi_text_only(self, question):
        prompt = (
            f"<|user|>\n{TEXT_ONLY_PROMPT.format(question=question)}<|end|>\n"
            f"<|assistant|>\n"
        )
        inputs = self.processor(text=prompt, return_tensors="pt")
        return self._generate(inputs)

    def _phi_with_image(self, image, text_prompt=None):
        prompt = (
            f"<|user|>\n<|image_1|>\n{text_prompt or IMAGE_PROMPT}<|end|>\n"
            f"<|assistant|>\n"
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        return self._generate(inputs)

    # ══════════════════════════════════════════════════════════════════════════
    #  MINICPM-V-2.6 (custom chat API)
    # ══════════════════════════════════════════════════════════════════════════

    def _minicpm_text_only(self, question):
        messages = [{"role": "user", "content": TEXT_ONLY_PROMPT.format(question=question)}]
        response = self.model.chat(
            image=None, msgs=messages, tokenizer=self.tokenizer,
            sampling=False, max_new_tokens=self.max_new_tokens,
        )
        return response.strip()

    def _minicpm_with_image(self, image, text_prompt=None):
        messages = [{"role": "user", "content": [
            image,
            text_prompt or IMAGE_PROMPT,
        ]}]
        response = self.model.chat(
            image=None, msgs=messages, tokenizer=self.tokenizer,
            sampling=False, max_new_tokens=self.max_new_tokens,
        )
        return response.strip()

    # ══════════════════════════════════════════════════════════════════════════
    #  IDEFICS3 (HuggingFace)
    # ══════════════════════════════════════════════════════════════════════════

    def _idefics_text_only(self, question):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": TEXT_ONLY_PROMPT.format(question=question)}
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt")
        return self._generate(inputs)

    def _idefics_with_image(self, image, text_prompt=None):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text_prompt or IMAGE_PROMPT},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        return self._generate(inputs)

    # ══════════════════════════════════════════════════════════════════════════
    #  SHARED GENERATION (for HuggingFace generate() models)
    # ══════════════════════════════════════════════════════════════════════════

    def _generate(self, inputs):
        """Standard HuggingFace generate() — used by all except InternVL2/MiniCPM."""
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        n = inputs["input_ids"].shape[1]
        decoded = self.processor.decode(output_ids[0][n:], skip_special_tokens=True)
        return decoded.strip()

    # ══════════════════════════════════════════════════════════════════════════
    #  CLEANUP
    # ══════════════════════════════════════════════════════════════════════════

    def unload(self):
        """Free GPU memory between model runs."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"Unloaded {self.model_name} — "
                  f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
