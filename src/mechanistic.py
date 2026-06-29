"""
Mechanistic investigation tools.

Provides interpretability analyses to understand WHY vision encoding
affects mathematical reasoning:

1. Attention map extraction and visualization
2. Token-level OCR comparison (what the model "reads" from images)
3. Hidden state probing (representation similarity between modalities)
4. Logit lens analysis (how predictions evolve through layers)
"""

import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENTION MAP EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_attention_maps(model, processor, image: Image.Image,
                           prompt: str, model_type: str = "qwen") -> Dict:
    """
    Extract attention maps from a VLM during image-based inference.

    Returns dict with:
      - attention_weights: list of (n_heads, seq_len, seq_len) per layer
      - token_labels: list of token strings
      - image_token_range: (start, end) indices for image tokens
    """
    if model_type == "qwen":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=prompt, images=[image], return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    # Extract attention weights
    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
    attention_maps = [attn[0].cpu().float().numpy() for attn in attentions]

    # Decode tokens
    input_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = [processor.decode([tid]) for tid in input_ids]

    # Find image token range (model-specific)
    image_token_ids = set()
    if hasattr(processor, "image_token_id"):
        image_token_ids.add(processor.image_token_id)

    img_start, img_end = -1, -1
    for i, tid in enumerate(input_ids):
        if tid in image_token_ids or tokens[i].strip() in ["<image>", "<|image|>", "<|vision_start|>"]:
            if img_start == -1:
                img_start = i
            img_end = i + 1

    return {
        "attention_maps": attention_maps,
        "tokens": tokens,
        "image_token_range": (img_start, img_end),
        "n_layers": len(attention_maps),
        "n_heads": attention_maps[0].shape[0] if attention_maps else 0,
        "seq_len": len(tokens),
    }


def plot_attention_to_image(attention_data: Dict, layer: int = -1,
                            head: int = None, save_path: str = None):
    """
    Plot how much attention text tokens pay to image tokens.

    Args:
        attention_data: Output from extract_attention_maps
        layer: Which layer to visualize (-1 = last)
        head: Which head (None = average across all heads)
    """
    attn = attention_data["attention_maps"][layer]  # (heads, seq, seq)
    tokens = attention_data["tokens"]
    img_start, img_end = attention_data["image_token_range"]

    if img_start == -1:
        print("No image tokens found in input")
        return

    if head is not None:
        attn_matrix = attn[head]  # (seq, seq)
    else:
        attn_matrix = attn.mean(axis=0)  # average across heads

    # For each text token, sum attention to image tokens
    n_tokens = len(tokens)
    text_to_image_attn = np.zeros(n_tokens)
    for i in range(n_tokens):
        if img_start <= i < img_end:
            continue  # skip image tokens
        text_to_image_attn[i] = attn_matrix[i, img_start:img_end].sum()

    # Filter to text tokens only
    text_indices = [i for i in range(n_tokens) if i < img_start or i >= img_end]
    text_labels = [tokens[i][:15] for i in text_indices]
    text_attn = [text_to_image_attn[i] for i in text_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(max(12, len(text_labels) * 0.3), 5))
    ax.bar(range(len(text_labels)), text_attn, color="#C44E52", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(text_labels)))
    ax.set_xticklabels(text_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Attention to Image Tokens")
    head_str = f"head {head}" if head is not None else "avg all heads"
    ax.set_title(f"Text → Image Attention (layer {layer}, {head_str})")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cross_modal_attention_heatmap(attention_data: Dict, layer: int = -1,
                                       save_path: str = None):
    """Heatmap of attention across all heads at a given layer."""
    attn = attention_data["attention_maps"][layer]  # (heads, seq, seq)
    img_start, img_end = attention_data["image_token_range"]
    n_heads = attn.shape[0]

    if img_start == -1:
        print("No image tokens found")
        return

    # For each head, compute total text→image and image→text attention
    text_to_img = np.zeros(n_heads)
    img_to_text = np.zeros(n_heads)
    n_text = attn.shape[1] - (img_end - img_start)

    for h in range(n_heads):
        # Text tokens attending to image tokens
        text_indices = list(range(img_start)) + list(range(img_end, attn.shape[1]))
        for t in text_indices:
            text_to_img[h] += attn[h, t, img_start:img_end].sum()
        text_to_img[h] /= max(len(text_indices), 1)

        # Image tokens attending to text tokens
        for i in range(img_start, min(img_end, attn.shape[1])):
            for t in text_indices:
                img_to_text[h] += attn[h, i, t]
        img_to_text[h] /= max(img_end - img_start, 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_heads)
    w = 0.35
    ax.bar(x - w / 2, text_to_img, w, label="Text → Image", color="#4C72B0")
    ax.bar(x + w / 2, img_to_text, w, label="Image → Text", color="#55A868")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Mean Cross-Modal Attention")
    ax.set_title(f"Cross-Modal Attention by Head (Layer {layer})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  TOKEN-LEVEL OCR COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_ocr_output(model, processor, question: str, image: Image.Image,
                       model_type: str = "qwen") -> Dict:
    """
    Compare what the model generates when asked to simply read/transcribe
    the image vs the original text.

    This isolates OCR quality from reasoning quality.
    """
    # Ask model to transcribe the image
    transcribe_prompt = (
        "Read the text in this image and write it out exactly as written. "
        "Do not solve anything, just transcribe the text word for word."
    )

    if model_type == "qwen":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": transcribe_prompt},
        ]}]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    else:
        inputs = processor(
            text=f"[INST] <image>\n{transcribe_prompt} [/INST]",
            images=[image], return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=512, do_sample=False)

    n = inputs["input_ids"].shape[1]
    transcription = processor.decode(output_ids[0][n:], skip_special_tokens=True).strip()

    # Compare transcription to original
    original_words = set(question.lower().split())
    transcribed_words = set(transcription.lower().split())

    # Extract numbers specifically
    original_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", question))
    transcribed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", transcription))

    return {
        "original": question,
        "transcription": transcription,
        "word_overlap": len(original_words & transcribed_words) / max(len(original_words), 1),
        "numbers_original": original_numbers,
        "numbers_transcribed": transcribed_numbers,
        "numbers_correct": original_numbers == transcribed_numbers,
        "numbers_missing": original_numbers - transcribed_numbers,
        "numbers_hallucinated": transcribed_numbers - original_numbers,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HIDDEN STATE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def compare_hidden_states(model, processor, question: str, image: Image.Image,
                          model_type: str = "qwen") -> Dict:
    """
    Compare internal representations between text-only and image+text conditions.

    Extracts hidden states from the last layer and computes cosine similarity
    to measure how different the model's internal state is across modalities.
    """
    # Text-only forward pass
    if model_type == "qwen":
        messages_text = [{"role": "user", "content": [
            {"type": "text", "text": f"Solve: {question}"}]}]
        prompt_text = processor.apply_chat_template(
            messages_text, tokenize=False, add_generation_prompt=True)
        inputs_text = processor(text=[prompt_text], return_tensors="pt")

        messages_img = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Solve the math problem in the image."}]}]
        prompt_img = processor.apply_chat_template(
            messages_img, tokenize=False, add_generation_prompt=True)
        inputs_img = processor(text=[prompt_img], images=[image], return_tensors="pt")
    else:
        inputs_text = processor(text=f"Solve: {question}", return_tensors="pt")
        inputs_img = processor(text="<image>\nSolve the math problem.",
                               images=[image], return_tensors="pt")

    inputs_text = {k: v.to(model.device) for k, v in inputs_text.items()}
    inputs_img = {k: v.to(model.device) for k, v in inputs_img.items()}

    with torch.no_grad():
        out_text = model(**inputs_text, output_hidden_states=True, return_dict=True)
        out_img = model(**inputs_img, output_hidden_states=True, return_dict=True)

    # Get last hidden state, mean-pooled
    hs_text = out_text.hidden_states[-1][0].mean(dim=0).cpu().float()
    hs_img = out_img.hidden_states[-1][0].mean(dim=0).cpu().float()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        hs_text.unsqueeze(0), hs_img.unsqueeze(0)).item()

    # L2 distance
    l2_dist = torch.norm(hs_text - hs_img).item()

    return {
        "cosine_similarity": cos_sim,
        "l2_distance": l2_dist,
        "text_norm": torch.norm(hs_text).item(),
        "image_norm": torch.norm(hs_img).item(),
    }


def batch_hidden_state_analysis(model, processor, questions: List[str],
                                 images: List[Image.Image],
                                 model_type: str = "qwen",
                                 max_problems: int = 50) -> pd.DataFrame:
    """Run hidden state comparison on a batch of problems."""
    import pandas as pd
    from tqdm import tqdm

    results = []
    for i, (q, img) in enumerate(tqdm(
            zip(questions[:max_problems], images[:max_problems]),
            total=min(len(questions), max_problems), desc="Hidden states")):
        try:
            r = compare_hidden_states(model, processor, q, img, model_type)
            r["problem_id"] = i
            results.append(r)
        except Exception as e:
            results.append({"problem_id": i, "error": str(e)})

    return pd.DataFrame(results)
