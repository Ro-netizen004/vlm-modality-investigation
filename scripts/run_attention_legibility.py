#!/usr/bin/env python3
"""
Phase 7 — Mechanistic (attention under legibility).

Complements the Phase 6 behavioral legibility curve with a *mechanistic* one:
for the mismatch input (image of problem i + text of problem i+1) at each image
corruption level, measure how much the text tokens attend to the image tokens.

Why this exists: the behavioral text-preference is bounded at ~100% for most
models (a ceiling), so it can't reveal whether preference tracks legibility.
Mean text->image attention is a *continuous* signal with full dynamic range, so
it can show the model down-weighting (or not) the image as it degrades even when
the behavioral preference is pinned.

  Attention DROPS as the image degrades  -> reliability-aware down-weighting
  Attention FLAT  as the image degrades  -> fixed attention, insensitive to legibility

Scope: reliable only on the Qwen family (eager attention + detectable image-token
range via extract_attention_maps). Extending to Phi/InternVL needs per-architecture
image-token handling in src/mechanistic.py — a follow-up, not done here.

Usage:
    python scripts/run_attention_legibility.py --smoke   # 1 problem, 1 level, sanity check
    python scripts/run_attention_legibility.py --benchmark gsm8k --noise-levels 0 2 4 5
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import VLMModel
from src.noise import NOISE_LEVELS, apply_noise_to_images, render_noisy_images
from src.mechanistic import extract_attention_maps
from src.benchmarks import load_benchmark

# Only the Qwen family reliably exposes attentions + a detectable image-token range.
ATTENTION_MODELS = {
    "Qwen2.5-VL-7B-Instruct": {"name": "Qwen/Qwen2.5-VL-7B-Instruct", "type": "qwen"},
    "Qwen2-VL-2B-Instruct":   {"name": "Qwen/Qwen2-VL-2B-Instruct",   "type": "qwen"},
}
DEFAULT_MODELS = list(ATTENTION_MODELS.keys())
TEXT_MATH_BENCHMARKS = ["gsm8k", "svamp", "math"]


def mismatch_prompt(text_question):
    return ("Solve the following math problem step by step. "
            "End with '#### <answer>'.\n\n"
            f"Problem: {text_question}")


def _atomic_write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def mean_text_to_image_attention(attn_data) -> float:
    """
    Scalar in ~[0,1]: mean attention mass from text tokens to image tokens,
    averaged across heads and layers. None if no image tokens were detected.
    """
    maps = attn_data["attention_maps"]           # list per layer of (heads, seq, seq)
    img_start, img_end = attn_data["image_token_range"]
    seq = attn_data["seq_len"]
    if img_start < 0 or img_end <= img_start or not maps:
        return None
    text_idx = [i for i in range(seq) if i < img_start or i >= img_end]
    if not text_idx:
        return None

    per_layer = []
    for layer_attn in maps:                      # (heads, seq, seq)
        m = layer_attn.mean(axis=0)              # avg over heads -> (seq, seq)
        # for each text token, total attention to the image-token span
        tti = m[np.ix_(text_idx, list(range(img_start, img_end)))].sum(axis=1)
        per_layer.append(float(tti.mean()))
    return float(np.mean(per_layer))


def run_level(model, processor, model_type, level, questions, images, image_dir, n, out_dir):
    config = NOISE_LEVELS[level]
    name = config["name"]
    final_path = os.path.join(out_dir, f"level_{level}_{name}.json")
    if os.path.exists(final_path):
        with open(final_path) as f:
            return json.load(f)

    level_img_dir = os.path.join(image_dir, f"level_{level}_{name}")
    vals = []
    n_no_img_tokens = 0
    for i in tqdm(range(n), desc=f"L{level}"):
        txt_idx = (i + 1) % n
        img_path = os.path.join(level_img_dir, f"q{i:03d}.png")
        try:
            img = Image.open(img_path).convert("RGB")
            attn = extract_attention_maps(
                model, processor, img, mismatch_prompt(questions[txt_idx]),
                model_type=model_type)
            score = mean_text_to_image_attention(attn)
            if score is None:
                n_no_img_tokens += 1
            else:
                vals.append(score)
        except Exception as e:
            print(f"  problem {i}: {e}")

    res = {
        "level": level,
        "name": name,
        "n_measured": len(vals),
        "n_no_image_tokens": n_no_img_tokens,
        "mean_text_to_image_attention": (float(np.mean(vals)) if vals else None),
        "std": (float(np.std(vals)) if vals else None),
    }
    _atomic_write_json(final_path, res)
    return res


def run_model(model_key, questions, images, image_dir, n, levels, out_root):
    mc = ATTENTION_MODELS[model_key]
    out_dir = os.path.join(out_root, model_key)
    os.makedirs(out_dir, exist_ok=True)

    remaining = [L for L in levels
                 if not os.path.exists(os.path.join(out_dir, f"level_{L}_{NOISE_LEVELS[L]['name']}.json"))]
    if not remaining:
        print(f"  {model_key}: all levels done")
    else:
        print(f"\n{'='*60}\n  {model_key}  (attention; levels: {remaining})\n{'='*60}")
        # eager attention is required for output_attentions to be populated.
        vlm = VLMModel(model_name=mc["name"], model_type=mc["type"],
                       torch_dtype="bfloat16", attn_implementation="eager")
        vlm.load()
        for level in levels:
            res = run_level(vlm.model, vlm.processor, mc["type"], level,
                            questions, images, image_dir, n, out_dir)
            a = res["mean_text_to_image_attention"]
            print(f"  L{level} {res['name']:18s}: "
                  f"attn_to_image={a:.4f}" if a is not None else
                  f"  L{level} {res['name']}: no measurable attention")
        vlm.unload()

    # summary across whatever levels exist
    by_level = {}
    for fn in os.listdir(out_dir):
        m = re.match(r"level_(\d+)_.*\.json$", fn)
        if m:
            with open(os.path.join(out_dir, fn)) as f:
                by_level[int(m.group(1))] = json.load(f)
    ordered = sorted(by_level)
    summary = {
        "model": model_key, "n": n,
        "attention_by_level": {L: by_level[L]["mean_text_to_image_attention"] for L in ordered},
    }
    _atomic_write_json(os.path.join(out_dir, "attention_summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 7 — attention under legibility")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmark", default="gsm8k", choices=TEXT_MATH_BENCHMARKS)
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--noise-levels", nargs="+", type=int, default=[0, 2, 4, 5])
    parser.add_argument("--noise-image-dir", default=None)
    parser.add_argument("--output-dir", default="results/phase7_attention")
    parser.add_argument("--smoke", action="store_true",
                        help="Sanity check: 1 problem, 1 level, first model only. "
                             "Run this on GAIVI before the full grid to confirm "
                             "attentions and the image-token range are detected.")
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.smoke:
        args.models = args.models[:1]
        args.num_problems = 1
        args.noise_levels = [args.noise_levels[0]]

    out_root = (args.output_dir if args.benchmark == "gsm8k"
                else os.path.join(args.output_dir, args.benchmark))
    os.makedirs(out_root, exist_ok=True)

    items = load_benchmark(args.benchmark, args.num_problems, use_hf=True)
    questions = [it.question for it in items]
    images = [it.image for it in items]
    n = len(questions)
    print(f"Loaded {n} {args.benchmark} problems (canonical HF images)")

    # Same noise images as Phase 6: applied on top of the canonical HF renders.
    image_dir = args.noise_image_dir or os.path.join(out_root, "noise_images")
    have_images = all(
        os.path.isdir(os.path.join(image_dir, f"level_{L}_{NOISE_LEVELS[L]['name']}"))
        for L in args.noise_levels)
    if not have_images:
        os.makedirs(image_dir, exist_ok=True)
        if all(img is not None for img in images):
            print(f"Applying noise to canonical HF images -> {image_dir} ...")
            apply_noise_to_images(images, image_dir,
                                  noise_levels=args.noise_levels, texts=questions)
        else:
            print("WARNING: some canonical images missing — falling back to fresh render.")
            render_noisy_images(questions, image_dir, noise_levels=args.noise_levels)

    all_summaries = {}
    for model_key in args.models:
        if model_key not in ATTENTION_MODELS:
            print(f"'{model_key}' not attention-supported (Qwen family only) — skipping")
            continue
        all_summaries[model_key] = run_model(
            model_key, questions, images, image_dir, n, args.noise_levels, out_root)

    _atomic_write_json(os.path.join(out_root, "attention_all.json"), all_summaries)
    print(f"\nPhase 7 attention complete ({args.benchmark}).")
    if args.smoke:
        print("SMOKE OK if 'attn_to_image' printed a number above (not 'no measurable "
              "attention'). If it was None, the image-token range wasn't detected — "
              "check extract_attention_maps for this architecture before the full run.")


if __name__ == "__main__":
    main()
