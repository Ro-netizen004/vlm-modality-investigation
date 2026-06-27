#!/usr/bin/env python3
"""
Phase 4 — Noise ablation runner (CLI version of Noise_Ablation.ipynb).

Runs the rendered-image (C2) condition across multiple noise levels on GSM8K,
plus a text-only baseline, for one or more models. Outputs per-level accuracy
with confidence intervals and error-type distributions.

Usage:
    python scripts/run_noise_ablation.py
    python scripts/run_noise_ablation.py --models Idefics3-8B-Llama3 InternVL2-8B MiniCPM-V-2_6
    python scripts/run_noise_ablation.py --num-problems 200 --output-dir results/phase4
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
from src.evaluation import (
    answers_match, classify_error, compute_accuracy, bootstrap_ci, binomial_ci,
)
from src.noise import (
    render_noisy_images,
    NOISE_LEVELS as NOISE_CONFIGS,
)

# Full model registry (HF id + loader type) — must match configs/default.yaml.
MODEL_REGISTRY = {
    "Qwen2-VL-2B-Instruct":          {"name": "Qwen/Qwen2-VL-2B-Instruct",          "type": "qwen"},
    "llava-v1.6-mistral-7b-hf":      {"name": "llava-hf/llava-v1.6-mistral-7b-hf",  "type": "llava"},
    "Qwen2.5-VL-7B-Instruct":        {"name": "Qwen/Qwen2.5-VL-7B-Instruct",        "type": "qwen"},
    "Idefics3-8B-Llama3":            {"name": "HuggingFaceM4/Idefics3-8B-Llama3",   "type": "idefics"},
    "MiniCPM-V-2_6":                 {"name": "openbmb/MiniCPM-V-2_6",              "type": "minicpm"},
    "InternVL2-8B":                  {"name": "OpenGVLab/InternVL2-8B",             "type": "internvl"},
    "llava-onevision-qwen2-7b-ov-hf":{"name": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "type": "llava_onevision"},
    "Phi-3.5-vision-instruct":       {"name": "microsoft/Phi-3.5-vision-instruct",  "type": "phi"},
}

# Default subset spans the resilience spectrum (vulnerable / middle / resilient).
DEFAULT_MODELS = ["Idefics3-8B-Llama3", "MiniCPM-V-2_6", "InternVL2-8B"]


def run_text_baseline(vlm, questions, references, out_path):
    n = len(questions)
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    correct = []
    for q, ref in tqdm(zip(questions, references), total=n, desc="Text"):
        try:
            pred = vlm.generate_text_only(q)
        except Exception as e:
            pred = f"ERROR: {e}"
        correct.append(answers_match(pred, ref))
    baseline = {
        "accuracy": compute_accuracy(correct),
        "ci_95": list(binomial_ci(sum(correct), n)),
        "n": n,
    }
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def run_noise_level(vlm, level, references, image_dir, out_path):
    n = len(references)
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)

    config = NOISE_CONFIGS[level]
    level_dir = os.path.join(image_dir, f"level_{level}_{config['name']}")

    correct, errors = [], []
    for i in tqdm(range(n), desc=f"L{level}"):
        img_path = os.path.join(level_dir, f"q{i:03d}.png")
        try:
            img = Image.open(img_path).convert("RGB")
            pred = vlm.generate_with_image(img)
        except Exception as e:
            pred = f"ERROR: {e}"
        correct.append(answers_match(pred, references[i]))
        errors.append(classify_error(pred, references[i]))

    result = {
        "level": level,
        "name": config["name"],
        "description": config["description"],
        "accuracy": compute_accuracy(correct),
        "error_distribution": dict(Counter(errors)),
        "ci_95": list(binomial_ci(sum(correct), n)),
        "boot_ci_95": list(bootstrap_ci(correct)),
        "n": n,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 4 noise ablation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model folder names from the registry")
    parser.add_argument("--num-problems", type=int, default=200,
                        help="GSM8K subset size (default 200)")
    parser.add_argument("--noise-levels", nargs="+", type=int,
                        default=list(range(10)), help="Noise levels 0-9")
    parser.add_argument("--output-dir", default="results/phase4")
    args = parser.parse_args()

    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.num_problems))
    questions = list(ds["question"])
    references = list(ds["answer"])
    n = len(questions)
    print(f"Loaded {n} GSM8K problems")

    # Render noisy images once (shared across all models)
    image_dir = os.path.join(args.output_dir, "images")
    print("Rendering noisy images at all requested levels...")
    render_noisy_images(questions, image_dir, noise_levels=args.noise_levels)

    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model '{model_key}' — skipping")
            continue

        mc = MODEL_REGISTRY[model_key]
        model_out = os.path.join(args.output_dir, model_key)
        os.makedirs(model_out, exist_ok=True)

        print(f"\n{'='*60}\n  {model_key}\n{'='*60}")
        vlm = VLMModel(model_name=mc["name"], model_type=mc["type"],
                       max_new_tokens=512, torch_dtype="bfloat16")
        vlm.load()

        baseline = run_text_baseline(
            vlm, questions, references,
            os.path.join(model_out, "text_baseline.json"))
        print(f"Text baseline: acc={baseline['accuracy']:.3f}")

        summary = {"model": model_key, "n": n,
                   "text_baseline": baseline["accuracy"], "levels": {}}

        for level in args.noise_levels:
            res = run_noise_level(
                vlm, level, references, image_dir,
                os.path.join(model_out, f"level_{level}_{NOISE_CONFIGS[level]['name']}.json"))
            drop = res["accuracy"] - baseline["accuracy"]
            summary["levels"][level] = {
                "name": res["name"],
                "accuracy": res["accuracy"],
                "drop_vs_text": drop,
            }
            print(f"  L{level} {res['name']:18s}: acc={res['accuracy']:.3f}  "
                  f"drop_vs_text={drop:+.3f}")

        with open(os.path.join(model_out, "noise_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        vlm.unload()

    print("\nNoise ablation complete.")


if __name__ == "__main__":
    main()
