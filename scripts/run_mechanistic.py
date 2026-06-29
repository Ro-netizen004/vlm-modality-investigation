#!/usr/bin/env python3
"""
Phase 6 — Mechanistic analysis runner (CLI version of Mechanistic_Analysis.ipynb).

Produces three complementary pieces of evidence for *why* text dominates:
  1. Attention-to-image: how much attention non-image tokens pay to image tokens,
     per layer, in both the rendered-image and mismatch conditions. If the model
     attends little to the image (and even less under mismatch), that is direct
     evidence it relies on the text channel.
  2. OCR quality: can the model transcribe the rendered text at all? Isolates
     reading ability from reasoning ability.
  3. Representation similarity: cosine similarity between text-only and
     image+text hidden states — how aligned the two modalities' internal states are.

Compute-light: ~50 problems on 1-2 models finishes in well under an hour.

Usage:
    python scripts/run_mechanistic.py
    python scripts/run_mechanistic.py --models Qwen2-VL-2B-Instruct Qwen2.5-VL-7B-Instruct
    python scripts/run_mechanistic.py --num-problems 50 --output-dir results/phase6
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import VLMModel
from src.rendering import render_all_images, load_image
from src.mechanistic import (
    extract_attention_maps,
    plot_attention_to_image,
    plot_cross_modal_attention_heatmap,
    compare_ocr_output,
    batch_hidden_state_analysis,
)

# Only models with clean attention/hidden-state access are worth running here.
MODEL_REGISTRY = {
    "Qwen2-VL-2B-Instruct":   {"name": "Qwen/Qwen2-VL-2B-Instruct",   "type": "qwen"},
    "Qwen2.5-VL-7B-Instruct": {"name": "Qwen/Qwen2.5-VL-7B-Instruct", "type": "qwen"},
}
DEFAULT_MODELS = ["Qwen2-VL-2B-Instruct"]


def image_attention_fraction(attn_data):
    """
    For each layer, mean fraction of attention that non-image (text) query tokens
    pay to image key tokens. Returns a list of per-layer fractions, or None if the
    image-token range could not be located.
    """
    start, end = attn_data["image_token_range"]
    if start < 0 or end <= start:
        return None
    seq = attn_data["seq_len"]
    text_rows = [i for i in range(seq) if not (start <= i < end)]
    if not text_rows:
        return None
    per_layer = []
    for attn in attn_data["attention_maps"]:        # (heads, seq, seq)
        head_avg = attn.mean(axis=0)                # (seq, seq)
        # fraction of each text row's attention mass landing on image columns
        img_mass = head_avg[np.ix_(text_rows, range(start, end))].sum(axis=1)
        per_layer.append(float(img_mass.mean()))
    return per_layer


def mismatch_prompt(text_question):
    return ("Solve the following math problem step by step. "
            "End with '#### <answer>'.\n\n"
            f"Problem: {text_question}")


def run_model(model_key, questions, references, image_dir, n, out_root):
    mc = MODEL_REGISTRY[model_key]
    out_dir = os.path.join(out_root, model_key)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  {model_key}\n{'='*60}")

    vlm = VLMModel(model_name=mc["name"], model_type=mc["type"],
                   max_new_tokens=256, torch_dtype="bfloat16",
                   attn_implementation="eager")  # needed for output_attentions
    vlm.load()
    mt = mc["type"]
    summary = {"model": model_key, "n": n}

    # ── 1. Attention-to-image: rendered-image vs mismatch ──────────────────────
    solve_prompt = ("Read the math problem and solve it step by step. "
                    "End with '#### <answer>'.")
    rendered_layers, mismatch_layers = [], []
    for i in range(n):
        img = load_image(i, image_dir)
        try:
            a_rendered = extract_attention_maps(vlm.model, vlm.processor, img,
                                                solve_prompt, mt)
            f = image_attention_fraction(a_rendered)
            if f is not None:
                rendered_layers.append(f)
        except Exception as e:
            if i == 0:
                print(f"  [attention/rendered] error: {e}")
        try:
            txt = references_text = questions[(i + 1) % n]
            a_mismatch = extract_attention_maps(vlm.model, vlm.processor, img,
                                                mismatch_prompt(txt), mt)
            f = image_attention_fraction(a_mismatch)
            if f is not None:
                mismatch_layers.append(f)
        except Exception as e:
            if i == 0:
                print(f"  [attention/mismatch] error: {e}")

    if rendered_layers:
        summary["attn_image_fraction_rendered"] = np.mean(rendered_layers, axis=0).tolist()
    if mismatch_layers:
        summary["attn_image_fraction_mismatch"] = np.mean(mismatch_layers, axis=0).tolist()
    # Save a sample heatmap for the paper figure
    try:
        sample_img = load_image(0, image_dir)
        a = extract_attention_maps(vlm.model, vlm.processor, sample_img, solve_prompt, mt)
        plot_attention_to_image(a, layer=-1,
                                save_path=os.path.join(out_dir, "attn_to_image_lastlayer.png"))
        plot_cross_modal_attention_heatmap(a, layer=-1,
                                save_path=os.path.join(out_dir, "cross_modal_heatmap.png"))
    except Exception as e:
        print(f"  [attention/plot] error: {e}")

    # ── 2. OCR quality ─────────────────────────────────────────────────────────
    ocr_scores = []
    for i in range(min(n, 30)):
        try:
            r = compare_ocr_output(vlm.model, vlm.processor, questions[i],
                                   load_image(i, image_dir), mt)
            if "similarity" in r:
                ocr_scores.append(r["similarity"])
        except Exception as e:
            if i == 0:
                print(f"  [ocr] error: {e}")
    if ocr_scores:
        summary["ocr_similarity_mean"] = float(np.mean(ocr_scores))
        summary["ocr_n"] = len(ocr_scores)

    # ── 3. Representation similarity ───────────────────────────────────────────
    try:
        images = [load_image(i, image_dir) for i in range(min(n, 50))]
        hs_df = batch_hidden_state_analysis(vlm.model, vlm.processor,
                                            questions[:50], images, mt, max_problems=50)
        if "cosine_similarity" in hs_df.columns:
            vals = hs_df["cosine_similarity"].dropna()
            if len(vals):
                summary["hidden_cosine_mean"] = float(vals.mean())
                summary["hidden_cosine_std"] = float(vals.std())
        hs_df.to_csv(os.path.join(out_dir, "hidden_states.csv"), index=False)
    except Exception as e:
        print(f"  [hidden-states] error: {e}")

    with open(os.path.join(out_dir, "mechanistic_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {out_dir}/mechanistic_summary.json")
    if "attn_image_fraction_rendered" in summary:
        r = summary["attn_image_fraction_rendered"]
        print(f"  Attention to image (rendered, last layer): {r[-1]:.4f}")
    if "attn_image_fraction_mismatch" in summary:
        m = summary["attn_image_fraction_mismatch"]
        print(f"  Attention to image (mismatch, last layer): {m[-1]:.4f}")
    if "ocr_similarity_mean" in summary:
        print(f"  OCR similarity: {summary['ocr_similarity_mean']:.3f}")
    if "hidden_cosine_mean" in summary:
        print(f"  Hidden-state cosine sim: {summary['hidden_cosine_mean']:.3f}")

    vlm.unload()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 6 mechanistic analysis")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--output-dir", default="results/phase6")
    args = parser.parse_args()

    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.num_problems))
    questions = list(ds["question"])
    references = list(ds["answer"])
    n = len(questions)
    print(f"Loaded {n} GSM8K problems")

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    existing = sum(1 for f in os.listdir(image_dir) if f.endswith(".png"))
    if existing < n:
        print("Rendering images...")
        render_all_images(questions, image_dir)

    all_summaries = {}
    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model '{model_key}' — skipping (attention access not verified)")
            continue
        all_summaries[model_key] = run_model(model_key, questions, references,
                                             image_dir, n, args.output_dir)

    with open(os.path.join(args.output_dir, "mechanistic_all.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("\nMechanistic analysis complete.")


if __name__ == "__main__":
    main()
