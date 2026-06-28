#!/usr/bin/env python3
"""
Run error analysis across models and save combined report.

Usage:
    python scripts/run_error_analysis.py
    python scripts/run_error_analysis.py --models Idefics3-8B-Llama3 llava-v1.6-mistral-7b-hf
    python scripts/run_error_analysis.py --results-dir results
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.error_analysis import (
    analyze_disagreements,
    analyze_modality_gap_correlates,
    format_disagreement_report,
)

DEFAULT_MODELS = [
    "Idefics3-8B-Llama3",
    "llava-onevision-qwen2-7b-ov-hf",
    "llava-v1.6-mistral-7b-hf",
    "MiniCPM-V-2_6",
]


def run_model(model_dir: str) -> dict:
    model_name = os.path.basename(model_dir)
    csv_path = os.path.join(model_dir, "gsm8k_results.csv")
    if not os.path.exists(csv_path):
        print(f"  No gsm8k_results.csv — skipping {model_name}")
        return {}

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)

    disagree = analyze_disagreements(
        list(df.correct_text), list(df.correct_rendered),
        list(df.question), list(df.reference),
    )
    corr = analyze_modality_gap_correlates(
        list(df.correct_text), list(df.correct_rendered),
        list(df.question), list(df.reference),
    )

    report = format_disagreement_report(disagree)
    print(report)
    print("\nCorrelations with modality gap:")
    print(corr.to_string(index=False))

    # Save per-model report
    report_path = os.path.join(model_dir, "error_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"ERROR ANALYSIS — {model_name}\n\n")
        f.write(report)
        f.write("\n\nCorrelations with modality gap:\n")
        f.write(corr.to_string(index=False))
    print(f"\nSaved: {report_path}")

    # Save correlation CSV
    corr_path = os.path.join(model_dir, "error_analysis_correlations.csv")
    corr.to_csv(corr_path, index=False)

    return {
        "model": model_name,
        "summary": disagree["summary"],
        "group_stats": {k: v["stats"] for k, v in disagree.items() if k != "summary"},
        "correlations": corr.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run error analysis across models")
    parser.add_argument("--results-dir", default="results/phase1")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model folder names to analyze")
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        model_dir = os.path.join(args.results_dir, model_name)
        if not os.path.isdir(model_dir):
            print(f"Not found: {model_dir} — skipping")
            continue
        result = run_model(model_dir)
        if result:
            all_results[model_name] = result

    if not all_results:
        print("No results produced.")
        return

    # Cross-model summary
    print(f"\n\n{'='*60}")
    print("  CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<40} {'Agree%':>7} {'TextAdv':>8} {'ImgAdv':>8}")
    print(f"  {'-'*40} {'-'*7} {'-'*8} {'-'*8}")
    for model, r in all_results.items():
        s = r["summary"]
        print(f"  {model:<40} {s['agreement_rate']*100:>6.1f}% {s['text_advantage']:>8} {s['image_advantage']:>8}")

    # Save combined JSON
    out_path = os.path.join(args.results_dir, "error_analysis_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined summary saved to: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
