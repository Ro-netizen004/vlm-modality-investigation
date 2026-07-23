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

# Direct (no chain-of-thought) version of the MISMATCH prompt for conditional
# log-likelihood scoring: the prompt asks the TEXT problem while the (conflicting)
# image is attached — exactly the generation-time conflict, minus the reasoning.
# CLL(text_answer) vs CLL(image_answer) then measures which modality pulls the answer.
DIRECT_MISMATCH_PROMPT = (
    "Solve the following math problem. "
    "Give only the final numeric answer in the form '#### <answer>'.\n\n"
    "Problem: {q}"
)

# Role-counterbalancing control. The default prompt above labels the TEXT "Problem:"
# and never references the image, so it designates the text as the task -- confounding
# modality preference with instruction-following (degrading the text destroys the task
# statement; degrading the image only damages an unreferenced attachment). The neutral
# variant names both channels as interchangeable sources and designates neither, with
# the A/B assignment counterbalanced per item so source order cannot drive the result.
NEUTRAL_MISMATCH_PROMPT = (
    "You are given two sources describing a math problem. "
    "Source {img} is the attached image. Source {txt} is the text below.\n\n"
    "Source {txt}: {q}\n\n"
    "Give only the final numeric answer in the form '#### <answer>'."
)


def direct_mismatch_prompt(q, role="text_task", item_idx=0):
    """Direct-answer mismatch scaffold used for CLL scoring. See NEUTRAL_MISMATCH_PROMPT."""
    if role == "neutral":
        img, txt = ("A", "B") if item_idx % 2 == 0 else ("B", "A")
        return NEUTRAL_MISMATCH_PROMPT.format(img=img, txt=txt, q=q)
    return DIRECT_MISMATCH_PROMPT.format(q=q)

# Approx API prices ($/1M tokens) as (input, output) for usage-cost reporting.
# Keyed by the API model id (lowercased). Update when prices change.
API_PRICES = {
    "gpt-5.6-luna":          (1.00, 6.00),
    "gpt-4o":                (2.50, 10.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash":      (1.50, 9.00),
}


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
        # API-model token accounting (populated by openai/gemini calls)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        # Per-call output-token logprobs from the last API call (None for local models)
        self.last_logprobs = None
        self._openai_supports_logprobs = True  # flipped off if the model rejects logprobs

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
            "openai": self._load_openai,
            "gemini": self._load_gemini,
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

        # transformers >=4.48 removed DynamicCache.get_max_length (renamed to
        # get_max_cache_shape), but Phi-3.5's bundled modeling code still calls it
        # during generation. Restore it so generate() works. DynamicCache is
        # unbounded, so its "max length" is None.
        try:
            from transformers.cache_utils import DynamicCache
            if not hasattr(DynamicCache, "get_max_length"):
                if hasattr(DynamicCache, "get_max_cache_shape"):
                    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
                else:
                    DynamicCache.get_max_length = lambda self: None
        except Exception:
            pass

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
            "openai": self._openai_text_only,
            "gemini": self._gemini_text_only,
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
            "openai": self._openai_with_image,
            "gemini": self._gemini_with_image,
        }
        return dispatch[self.model_type](image, text_prompt)

    # ══════════════════════════════════════════════════════════════════════════
    #  OPENAI (API frontier models — no GPU; reads OPENAI_API_KEY from env)
    # ══════════════════════════════════════════════════════════════════════════

    def _load_openai(self):
        """Init the OpenAI client (no weights). self.model holds the client."""
        import os
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.model = OpenAI(api_key=api_key)

    @staticmethod
    def _pil_to_data_url(image):
        import base64, io
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _openai_chat(self, content):
        """One chat completion with exponential backoff; records usage + logprobs."""
        import time
        self.last_logprobs = None
        last_err = None
        base = dict(model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=max(self.max_new_tokens, 512))
        for attempt in range(6):
            try:
                r = self._openai_create(base)
                u = getattr(r, "usage", None)
                if u:
                    self.usage["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                    self.usage["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
                self.usage["calls"] += 1
                self.last_logprobs = self._extract_openai_logprobs(r)
                return r.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)  # 1,2,4,8,16,32s
        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

    def _openai_create(self, base):
        """Create a completion, dropping optional params the model rejects
        (temperature, logprobs) so unsupported models still return an answer."""
        opts = dict(base, temperature=0)
        if self._openai_supports_logprobs:
            opts.update(logprobs=True, top_logprobs=5)
        for _ in range(3):
            try:
                return self.model.chat.completions.create(**opts)
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in msg and "temperature" in opts:
                    opts.pop("temperature")
                elif "logprob" in msg and "logprobs" in opts:
                    opts.pop("logprobs", None); opts.pop("top_logprobs", None)
                    self._openai_supports_logprobs = False  # stop asking on later calls
                else:
                    raise
        return self.model.chat.completions.create(**opts)

    @staticmethod
    def _extract_openai_logprobs(r):
        """[{tok, lp, top:[{tok,lp}...]}, ...] for the generated tokens, or None."""
        try:
            content = r.choices[0].logprobs.content
        except Exception:
            return None
        if not content:
            return None
        out = []
        for t in content:
            top = [{"tok": a.token, "lp": a.logprob}
                   for a in (getattr(t, "top_logprobs", None) or [])]
            out.append({"tok": t.token, "lp": t.logprob, "top": top})
        return out

    def _openai_text_only(self, question):
        return self._openai_chat(
            [{"type": "text", "text": TEXT_ONLY_PROMPT.format(question=question)}])

    def _openai_with_image(self, image, text_prompt=None):
        return self._openai_chat([
            {"type": "text", "text": text_prompt or IMAGE_PROMPT},
            {"type": "image_url",
             "image_url": {"url": self._pil_to_data_url(image), "detail": "high"}},
        ])

    # ══════════════════════════════════════════════════════════════════════════
    #  GEMINI (Google API frontier models — no GPU; reads GEMINI_API_KEY)
    # ══════════════════════════════════════════════════════════════════════════

    def _load_gemini(self):
        """Init the Gemini client (no weights). self.model holds the client."""
        import os
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment.")
        self.model = genai.Client(api_key=api_key)

    def _gemini_generate(self, contents):
        """One generate_content call with backoff; records usage + logprobs."""
        import time
        from google.genai import types
        self.last_logprobs = None
        last_err = None
        for attempt in range(6):
            try:
                try:
                    r = self.model.models.generate_content(
                        model=self.model_name, contents=contents,
                        config=self._gemini_config(types, want_logprobs=True))
                except Exception as e:
                    if "logprob" in str(e).lower():  # model/SDK rejects logprobs
                        r = self.model.models.generate_content(
                            model=self.model_name, contents=contents,
                            config=self._gemini_config(types, want_logprobs=False))
                    else:
                        raise
                um = getattr(r, "usage_metadata", None)
                if um:
                    self.usage["input_tokens"] += getattr(um, "prompt_token_count", 0) or 0
                    self.usage["output_tokens"] += getattr(um, "candidates_token_count", 0) or 0
                self.usage["calls"] += 1
                self.last_logprobs = self._extract_gemini_logprobs(r)
                try:
                    return r.text or ""
                except Exception:
                    return ""  # blocked / no candidate → treated as empty prediction
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")

    def _gemini_config(self, types, want_logprobs):
        kw = dict(temperature=0, max_output_tokens=max(self.max_new_tokens, 512))
        if want_logprobs:
            try:
                return types.GenerateContentConfig(response_logprobs=True, logprobs=5, **kw)
            except TypeError:
                pass  # older SDK without logprob fields
        return types.GenerateContentConfig(**kw)

    @staticmethod
    def _extract_gemini_logprobs(r):
        """Best-effort logprob extraction; returns None on any schema mismatch
        (verified/adjusted after the smoke test)."""
        try:
            lr = r.candidates[0].logprobs_result
        except Exception:
            return None
        if lr is None:
            return None
        try:
            chosen = getattr(lr, "chosen_candidates", None) or []
            out = [{"tok": getattr(c, "token", None),
                    "lp": getattr(c, "log_probability", None)} for c in chosen]
            return out or None
        except Exception:
            return None

    def _gemini_text_only(self, question):
        return self._gemini_generate([TEXT_ONLY_PROMPT.format(question=question)])

    def _gemini_with_image(self, image, text_prompt=None):
        return self._gemini_generate([text_prompt or IMAGE_PROMPT, image.convert("RGB")])

    # ══════════════════════════════════════════════════════════════════════════
    #  USAGE REPORTING (API models)
    # ══════════════════════════════════════════════════════════════════════════

    def report_usage(self):
        """Print measured token usage + estimated cost for API models."""
        u = self.usage
        if not u["calls"]:
            return
        line = (f"[usage] {self.model_name}: {u['calls']} calls, "
                f"{u['input_tokens']:,} in + {u['output_tokens']:,} out tokens")
        price = API_PRICES.get(self.model_name.lower())
        if price:
            cost = u["input_tokens"] / 1e6 * price[0] + u["output_tokens"] / 1e6 * price[1]
            line += f"  ≈ ${cost:.4f}"
        else:
            line += "  (no price on file — tokens only)"
        print(line)

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
        """Standard HuggingFace generate() — used by all except InternVL2/MiniCPM.
        Captures per-token logprobs into self.last_logprobs (parallels the API paths)."""
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                output_scores=True,
                return_dict_in_generate=True,
            )
        n = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0][n:]
        self.last_logprobs = self._hf_logprobs(gen_ids, out.scores)
        decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
        return decoded.strip()

    def _hf_logprobs(self, gen_ids, scores, topk=5):
        """Per-generated-token logprobs (+ top-k alternatives) from generate() scores."""
        if not scores:
            return None
        out = []
        try:
            for t, logits in enumerate(scores):
                if t >= len(gen_ids):
                    break
                lp = torch.log_softmax(logits[0].float(), dim=-1)
                tok_id = int(gen_ids[t])
                vals, idx = torch.topk(lp, k=min(topk, lp.shape[-1]))
                out.append({
                    "tok": self._decode_tok(tok_id),
                    "lp": float(lp[tok_id]),
                    "top": [{"tok": self._decode_tok(int(i)), "lp": float(v)}
                            for v, i in zip(vals.tolist(), idx.tolist())],
                })
        except Exception:
            return out or None
        return out or None

    def _decode_tok(self, tok_id):
        dec = self.tokenizer or self.processor
        try:
            return dec.decode([tok_id])
        except Exception:
            return str(tok_id)

    # ══════════════════════════════════════════════════════════════════════════
    #  CONDITIONAL LOG-LIKELIHOOD (teacher-forced answer scoring; open models)
    # ══════════════════════════════════════════════════════════════════════════

    def _ctx_for_scoring(self, prompt_text):
        """Templated context string ending at the '#### ' answer position, per model
        type. Mirrors each type's generation prompt so scoring is on-distribution."""
        t = self.model_type
        if t in ("qwen", "llava_onevision"):
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt_text}]}]
            ctx = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        elif t == "idefics":
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt_text}]}]
            ctx = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        elif t == "llava":
            ctx = f"[INST] <image>\n{prompt_text} [/INST]"
        elif t == "phi":
            ctx = f"<|user|>\n<|image_1|>\n{prompt_text}<|end|>\n<|assistant|>\n"
        else:
            raise NotImplementedError(
                f"conditional_loglik not implemented for model_type={t}")
        return ctx + "#### "

    def _score_continuation(self, ctx_text, image, candidate):
        """Teacher-forced logprob of `candidate` appended to ctx_text (with image).
        Scores only the candidate tokens. Returns {sum, mean, n} or None."""
        tok = getattr(self.processor, "tokenizer", None) or self.tokenizer
        ids_ctx = tok(ctx_text, add_special_tokens=False)["input_ids"]
        ids_full = tok(ctx_text + candidate, add_special_tokens=False)["input_ids"]
        n_cand = len(ids_full) - len(ids_ctx)
        if n_cand <= 0:
            return None
        inputs = self.processor(text=ctx_text + candidate, images=[image],
                                return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0].float()  # [seq, vocab]
        ids = inputs["input_ids"][0]
        # logit at position p-1 predicts token at position p (teacher forcing).
        total = 0.0
        for p in range(len(ids) - n_cand, len(ids)):
            lp = torch.log_softmax(logits[p - 1], dim=-1)
            total += float(lp[int(ids[p])])
        return {"sum": total, "mean": total / n_cand, "n": n_cand}

    def conditional_loglik(self, image, candidate, text_question):
        """CLL of `candidate` under the direct MISMATCH scaffold (prompt asks the TEXT
        problem, conflicting image attached). Open _generate-family models only."""
        ctx = self._ctx_for_scoring(DIRECT_MISMATCH_PROMPT.format(q=text_question))
        return self._score_continuation(ctx, image, str(candidate))

    def arbitration_margin(self, image, text_answer, image_answer, text_question,
                           role="text_task", item_idx=0):
        """margin = CLL(text_answer) - CLL(image_answer), per-token normalized, scored
        under the mismatch scaffold. Positive = model favors the TEXT answer.

        role="neutral" swaps in the counterbalanced two-source scaffold so the CLL
        measure matches the generation-side framing (both must use the same role or
        the two measures answer different questions).
        """
        prompt = direct_mismatch_prompt(text_question, role, item_idx)
        return self.candidate_margin(image, text_answer, image_answer, prompt)

    def candidate_margin(self, image, text_answer, image_answer, prompt):
        """Score two answer candidates under an arbitrary shared multimodal prompt.

        Positive margin means the text-supported candidate has greater mean answer-token
        log probability than the image-supported candidate. This public wrapper supports
        conflict controls whose scaffold differs from the rendered-math mismatch prompt.
        """
        ctx = self._ctx_for_scoring(prompt)
        t = self._score_continuation(ctx, image, str(text_answer))
        im = self._score_continuation(ctx, image, str(image_answer))
        if t is None or im is None:
            return None
        return {
            "cll_text_mean": t["mean"], "cll_image_mean": im["mean"],
            "cll_text_sum": t["sum"], "cll_image_sum": im["sum"],
            "margin_mean": t["mean"] - im["mean"],
            "margin_sum": t["sum"] - im["sum"],
        }

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
