#!/usr/bin/env python3
"""
Legibility experiment — modality preference under image degradation.

Runs the MISMATCH condition (image of problem i + text of problem i+1) at each
image-corruption level, measuring how text preference changes as the image
becomes less legible. A flat curve indicates a fixed text bias insensitive to
input quality; a sloped curve indicates rational, reliability-aware arbitration.

Reuses the noisy images already rendered by the Phase 4 noise ablation
(results/phase4/images/level_<L>_<name>/q###.png), so no re-rendering is needed
if those exist.

Usage:
    python scripts/run_legibility.py
    python scripts/run_legibility.py --models Qwen2-VL-2B-Instruct Idefics3-8B-Llama3
    python scripts/run_legibility.py --num-problems 200 --noise-image-dir results/phase4/images
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import VLMModel
from src.evaluation import score_mismatch_follows
from src.noise import NOISE_LEVELS, render_noisy_images

# Same registry/types as the noise runner — must match configs/default.yaml.
MODEL_REGISTRY = {
    "Qwen2-VL-2B-Instruct":          {"name": "Qwen/Qwen2-VL-2B-Instruct",          "type": "qwen"},
    "llava-v1.6-mistral-7b-hf":      {"name": "llava-hf/llava-v1.6-mistral-7b-hf",  "type": "llava"},
    "Qwen2.5-VL-7B-Instruct":        {"name": "Qwen/Qwen2.5-VL-7B-Instruct",        "type": "qwen"},
    "Idefics3-8B-Llama3":            {"name": "HuggingFaceM4/Idefics3-8B-Llama3",   "type": "idefics"},
    "MiniCPM-V-2_6":                 {"name": "openbmb/MiniCPM-V-2_6",              "type": "minicpm"},
    "InternVL2-8B":                  {"name": "OpenGVLab/InternVL2-8B",             "type": "internvl"},
    "llava-onevision-qwen2-7b-ov-hf":{"name": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "type": "llava_onevision"},
}
DEFAULT_MODELS = ["Idefics3-8B-Llama3", "Qwen2.5-VL-7B-Instruct"]  # vulnerable + resilient anchors


def mismatch_prompt(text_question):
    return ("Solve the following math problem step by step. "
            "End with '#### <answer>'.\n\n"
            f"Problem: {text_question}")


def run_level(vlm, level, questions, references, image_dir, n):
    """Run the mismatch condition for all problems at one noise level."""
    config = NOISE_LEVELS[level]
    level_dir = os.path.join(image_dir, f"level_{level}_{config['name']}")
    follows = []
    for i in tqdm(range(n), desc=f"L{level}"):
        txt_idx = (i + 1) % n
        img_path = os.path.join(level_dir, f"q{i:03d}.png")
        try:
            img = Image.open(img_path).convert("RGB")
            pred = vlm.generate_with_image(img, text_prompt=mismatch_prompt(questions[txt_idx]))
        except Exception as e:
            pred = f"ERROR: {e}"
        follows.append(score_mismatch_follows(pred, references[i], references[txt_idx]))

    counts = Counter(follows)
    n_img = counts.get("image", 0)
    n_txt = counts.get("text", 0)
    decidable = n_img + n_txt
    return {
        "level": level,
        "name": config["name"],
        "counts": dict(counts),
        "decidable": decidable,
        "text_preference": (n_txt / decidable) if decidable else None,
        "image_preference": (n_img / decidable) if decidable else None,
        "neither_rate": counts.get("neither", 0) / n if n else None,
    }


def run_model(model_key, questions, references, image_dir, n, levels, out_root):
    mc = MODEL_REGISTRY[model_key]
    out_dir = os.path.join(out_root, model_key)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  {model_key}\n{'='*60}")

    vlm = VLMModel(model_name=mc["name"], model_type=mc["type"],
                   max_new_tokens=256, torch_dtype="bfloat16")
    vlm.load()

    by_level = {}
    for level in levels:
        res = run_level(vlm, level, questions, references, image_dir, n)
        by_level[level] = res
        tp = res["text_preference"]
        print(f"  L{level} {res['name']:18s}: text_pref={tp:.3f}  "
              f"decidable={res['decidable']}  neither={res['neither_rate']*100:.1f}%"
              if tp is not None else f"  L{level} {res['name']}: no decidable trials")
        # cache each level
        with open(os.path.join(out_dir, f"level_{level}_{res['name']}.json"), "w") as f:
            json.dump(res, f, indent=2)

    summary = {
        "model": model_key, "n": n,
        "text_preference_by_level": {L: by_level[L]["text_preference"] for L in levels},
        "neither_rate_by_level": {L: by_level[L]["neither_rate"] for L in levels},
    }
    with open(os.path.join(out_dir, "legibility_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_dir}/legibility_summary.json")

    # The headline: does text preference change across legibility?
    tps = [by_level[L]["text_preference"] for L in levels
           if by_level[L]["text_preference"] is not None]
    if tps:
        print(f"  Text preference range across levels: {min(tps):.3f}--{max(tps):.3f} "
              f"(spread {max(tps)-min(tps):.3f})")

    vlm.unload()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Legibility experiment (mismatch x noise)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--noise-levels", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--noise-image-dir", default="results/phase4/images",
                        help="Where the Phase 4 noisy images live (reused if present)")
    parser.add_argument("--output-dir", default="results/phase6_legibility")
    args = parser.parse_args()

    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.num_problems))
    questions = list(ds["question"])
    references = list(ds["answer"])
    n = len(questions)
    print(f"Loaded {n} GSM8K problems")

    # Reuse Phase 4 noisy images if available; otherwise render them.
    image_dir = args.noise_image_dir
    needed = all(
        os.path.isdir(os.path.join(image_dir, f"level_{L}_{NOISE_LEVELS[L]['name']}"))
        for L in args.noise_levels
    )
    if not needed:
        print(f"Noisy images not found in {image_dir} — rendering them...")
        os.makedirs(image_dir, exist_ok=True)
        render_noisy_images(questions, image_dir, noise_levels=args.noise_levels)
    else:
        print(f"Reusing existing noisy images from {image_dir}")

    all_summaries = {}
    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model '{model_key}' — skipping")
            continue
        all_summaries[model_key] = run_model(
            model_key, questions, references, image_dir, n,
            args.noise_levels, args.output_dir)

    with open(os.path.join(args.output_dir, "legibility_all.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("\nLegibility experiment complete.")


if __name__ == "__main__":
    main()
