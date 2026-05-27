#!/usr/bin/env python3
"""CLI entry point for VLM math benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gsm8k_experiment.config import DEFAULT_CONFIG, finalize_config
from gsm8k_experiment.datasets import load_dataset
from gsm8k_experiment.experiments.runner import run_experiment, summarize_results
from gsm8k_experiment.models import load_model
from gsm8k_experiment.visualize import plot_results


def parse_args():
    p = argparse.ArgumentParser(description="Run VLM math benchmark experiment")
    p.add_argument("--dataset-type", choices=["gsm8k", "svamp"], default=DEFAULT_CONFIG["dataset_type"])
    p.add_argument("--model-id", default=DEFAULT_CONFIG["model_id"])
    p.add_argument("--mode", choices=["text_only", "image_only", "text_and_image", "mismatch"], default=DEFAULT_CONFIG["experiment_mode"])
    p.add_argument("--num-problems", type=int, default=DEFAULT_CONFIG["num_problems"])
    p.add_argument("--metadata-csv", default=DEFAULT_CONFIG["metadata_csv"])
    p.add_argument("--image-root", default=DEFAULT_CONFIG["image_root"])
    p.add_argument("--hf-dataset-id", default=None, help="Override Hugging Face dataset repo id")
    p.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = finalize_config(
        {
            **DEFAULT_CONFIG,
            "dataset_type": args.dataset_type,
            "model_id": args.model_id,
            "experiment_mode": args.mode,
            "num_problems": args.num_problems,
            "metadata_csv": args.metadata_csv,
            "image_root": args.image_root,
            "output_dir": args.output_dir,
            "use_4bit": not args.no_4bit,
        }
    )
    if args.hf_dataset_id:
        cfg["hf_dataset_id"] = args.hf_dataset_id

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k:<20}: {v}")
    print("=" * 60)

    samples = load_dataset(cfg)
    processor, model = load_model(cfg)
    df = run_experiment(cfg, processor, model, samples)
    summary = summarize_results(cfg, df)

    df.to_csv(cfg["results_csv"], index=False)
    with open(cfg["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Accuracy: {summary['accuracy']:.2%} ({summary['n_correct']}/{summary['num_problems']})")
    print(f"Results: {cfg['results_csv']}")
    print(f"Summary: {cfg['summary_json']}")

    if not args.no_plots:
        plot_results(cfg, df, summary)


if __name__ == "__main__":
    main()
