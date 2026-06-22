#!/usr/bin/env python3
"""
Main benchmark runner — evaluates VLMs on GSM8K across modality conditions.

Usage:
    python scripts/run_benchmark.py                          # full GSM8K, all models
    python scripts/run_benchmark.py --num-problems 100       # first 100 only
    python scripts/run_benchmark.py --config configs/custom.yaml
    python scripts/run_benchmark.py --models "Qwen/Qwen2-VL-2B-Instruct"
    python scripts/run_benchmark.py --hf-images              # use canonical HF v2 images
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    answers_match, classify_error, compute_accuracy, compute_all_statistics,
    format_statistics_report, score_mismatch_follows,
)
from src.models import VLMModel
from src.rendering import load_image, render_all_images
from src.visualization import (
    plot_accuracy_comparison, plot_error_breakdown,
    plot_mismatch_dominance, plot_statistical_summary,
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_gsm8k(dataset_name, dataset_config, split, num_problems=None):
    full = load_dataset(dataset_name, dataset_config, split=split)
    if num_problems:
        full = full.select(range(min(num_problems, len(full))))
    print(f"Loaded {len(full)} problems from {dataset_name} {split} split")
    return full["question"], full["answer"]


def download_hf_images(image_dir: str, num_problems: int = None) -> None:
    """Download canonical v2 images from HuggingFace dataset instead of re-rendering locally."""
    os.makedirs(image_dir, exist_ok=True)
    hf = load_dataset("vlm-modality-research/gsm8k-rendered-vlm-v2", split="train")
    if num_problems:
        hf = hf.select(range(min(num_problems, len(hf))))
    existing = sum(1 for f in os.listdir(image_dir) if f.endswith(".png"))
    if existing >= len(hf):
        print(f"HF images already downloaded ({existing} found) — skipping")
        return
    print(f"Downloading {len(hf)} canonical images from HuggingFace...")
    for i, row in enumerate(tqdm(hf, desc="Images")):
        path = os.path.join(image_dir, f"q{i:03d}.png")
        if not os.path.exists(path):
            row["image"].save(path)
    print(f"Done — {len(hf)} images saved to {image_dir}")


def run_condition_text_only(model, questions, references):
    """Condition 1: text-only baseline."""
    predictions, correct_flags, errors = [], [], []
    for q, ref in tqdm(zip(questions, references), total=len(questions), desc="Text-Only"):
        try:
            pred = model.generate_text_only(q)
        except Exception as e:
            pred = f"ERROR: {e}"
        predictions.append(pred)
        correct_flags.append(answers_match(pred, ref))
        errors.append(classify_error(pred, ref))
    return predictions, correct_flags, errors


def run_condition_rendered_image(model, questions, references, image_dir):
    """Condition 2: rendered image."""
    predictions, correct_flags, errors = [], [], []
    for i, (q, ref) in enumerate(tqdm(zip(questions, references),
                                       total=len(questions), desc="Rendered Image")):
        try:
            img = load_image(i, image_dir)
            pred = model.generate_with_image(img)
        except Exception as e:
            pred = f"ERROR: {e}"
        predictions.append(pred)
        correct_flags.append(answers_match(pred, ref))
        errors.append(classify_error(pred, ref))
    return predictions, correct_flags, errors


def run_condition_mismatch(model, questions, references, image_dir):
    """Condition 3: modality mismatch (image_i + text_{i+1})."""
    n = len(questions)
    predictions, follows_list = [], []

    for i in tqdm(range(n), desc="Mismatch"):
        txt_idx = (i + 1) % n
        mismatch_prompt = (
            f"Solve the following math problem step by step. "
            f"End with '#### <answer>'.\n\n"
            f"Problem: {questions[txt_idx]}"
        )
        try:
            img = load_image(i, image_dir)
            pred = model.generate_with_image(img, text_prompt=mismatch_prompt)
        except Exception as e:
            pred = f"ERROR: {e}"

        follows = score_mismatch_follows(pred, references[i], references[txt_idx])
        predictions.append(pred)
        follows_list.append(follows)

    return predictions, follows_list


def save_results(model_name, questions, references, output_dir,
                 text_preds, text_correct, text_errors,
                 img_preds, img_correct, img_errors,
                 mm_preds, mm_follows,
                 stats_dict):
    """Save all results to CSV and statistics to JSON/text."""
    short = model_name.split("/")[-1]
    model_dir = os.path.join(output_dir, short)
    os.makedirs(model_dir, exist_ok=True)

    n = len(questions)

    results_df = pd.DataFrame({
        "problem_id": list(range(n)),
        "question": questions,
        "reference": references,
        "pred_text": text_preds,
        "correct_text": text_correct,
        "error_text": text_errors,
        "pred_rendered": img_preds,
        "correct_rendered": img_correct,
        "error_rendered": img_errors,
        "pred_mismatch": mm_preds,
        "mismatch_follows": mm_follows,
    })
    results_df.to_csv(os.path.join(model_dir, "gsm8k_results.csv"), index=False)

    # Error summary
    from src.evaluation import error_counts
    categories = ["correct", "arithmetic_error", "reasoning_error", "no_number", "vision_error"]
    summary = pd.DataFrame([
        {"Condition": "Text-Only", **error_counts(text_errors, categories)},
        {"Condition": "Rendered Image", **error_counts(img_errors, categories)},
    ])
    summary.to_csv(os.path.join(model_dir, "error_summary.csv"), index=False)

    # Mismatch results
    mm_df = pd.DataFrame({
        "problem_id": list(range(n)),
        "image_question": questions,
        "image_reference": references,
        "text_question": [questions[(i + 1) % n] for i in range(n)],
        "text_reference": [references[(i + 1) % n] for i in range(n)],
        "prediction": mm_preds,
        "follows": mm_follows,
    })
    mm_df.to_csv(os.path.join(model_dir, "mismatch_results.csv"), index=False)

    # Disagreements
    disagree = results_df[results_df["correct_text"] != results_df["correct_rendered"]]
    disagree.to_csv(os.path.join(model_dir, "disagreements.csv"), index=False)

    # Statistics
    with open(os.path.join(model_dir, "statistics.json"), "w") as f:
        serializable = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in stats_dict.items()}
        json.dump(serializable, f, indent=2)

    report = format_statistics_report(stats_dict)
    with open(os.path.join(model_dir, "statistics_report.txt"), "w") as f:
        f.write(report)
    print(report)

    # Plots
    errors_by_cond = {"Text-Only": text_errors, "Rendered Image": img_errors}
    plot_error_breakdown(errors_by_cond, model_name, model_dir)
    plot_mismatch_dominance(Counter(mm_follows), model_name, model_dir)

    print(f"\nAll results saved to: {model_dir}")
    return stats_dict


def main():
    parser = argparse.ArgumentParser(description="VLM Modality Benchmark")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num-problems", type=int, default=None,
                        help="Override number of problems (null = full test set)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip image rendering (use existing images)")
    parser.add_argument("--hf-images", action="store_true",
                        help="Download canonical v2 images from HuggingFace instead of rendering locally")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config.get("seed", 42))

    num_problems = args.num_problems or config["dataset"].get("num_problems")
    output_dir = args.output_dir or config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    questions, references = load_gsm8k(
        config["dataset"]["name"],
        config["dataset"]["config"],
        config["dataset"]["split"],
        num_problems,
    )
    questions = list(questions)
    references = list(references)
    n = len(questions)
    print(f"Running benchmark on {n} problems\n")

    # Images — either download canonical HF v2 or render locally
    image_dir = os.path.join(output_dir, "rendered_images")
    if args.hf_images:
        download_hf_images(image_dir, num_problems)
    elif not args.skip_render:
        render_all_images(questions, image_dir, config.get("image_rendering", {}))

    # Select models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        model_configs = [m for m in config["models"] if m["name"] in model_names]
        if not model_configs:
            model_configs = [{"name": m, "type": "qwen", "max_new_tokens": 256,
                              "torch_dtype": "bfloat16", "quantize": None}
                             for m in model_names]
    else:
        model_configs = config["models"]

    all_stats = {}

    for mc in model_configs:
        short = mc["name"].split("/")[-1]
        model_dir = os.path.join(output_dir, short)
        if os.path.exists(os.path.join(model_dir, "statistics.json")):
            print(f"\n>>> SKIPPING {short} — results already exist <<<")
            with open(os.path.join(model_dir, "statistics.json")) as f:
                all_stats[mc["name"]] = json.load(f)
            continue

        print(f"\n{'=' * 70}")
        print(f"  MODEL: {mc['name']}")
        print(f"{'=' * 70}\n")

        vlm = VLMModel(
            model_name=mc["name"],
            model_type=mc["type"],
            max_new_tokens=mc.get("max_new_tokens", 256),
            torch_dtype=mc.get("torch_dtype", "bfloat16"),
            quantize=mc.get("quantize"),
        )
        vlm.load()

        t0 = time.time()

        # Run all conditions
        text_preds, text_correct, text_errors = run_condition_text_only(
            vlm, questions, references)

        img_preds, img_correct, img_errors = run_condition_rendered_image(
            vlm, questions, references, image_dir)

        mm_preds, mm_follows = run_condition_mismatch(
            vlm, questions, references, image_dir)

        elapsed = time.time() - t0
        print(f"\nDominance: {Counter(mm_follows)}")
        print(f"\nTotal inference time: {elapsed / 60:.1f} minutes")

        # Statistics
        stats = compute_all_statistics(text_correct, img_correct, mm_follows)
        stats["elapsed_minutes"] = round(elapsed / 60, 1)

        # Save everything
        save_results(
            mc["name"], questions, references, output_dir,
            text_preds, text_correct, text_errors,
            img_preds, img_correct, img_errors,
            mm_preds, mm_follows,
            stats,
        )

        all_stats[mc["name"]] = stats
        vlm.unload()

    # Cross-model summary
    if len(all_stats) > 1:
        plot_accuracy_comparison(all_stats, output_dir)
        plot_statistical_summary(all_stats, output_dir)

        summary_path = os.path.join(output_dir, "cross_model_summary.json")
        serializable = {}
        for m, s in all_stats.items():
            serializable[m] = {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in s.items()}
        with open(summary_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nCross-model summary saved to: {summary_path}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
