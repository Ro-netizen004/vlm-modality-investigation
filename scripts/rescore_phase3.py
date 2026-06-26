#!/usr/bin/env python3
"""
Rescore Phase 3 results from existing CSVs using fixed answer matching.

The model predictions in the CSVs are correct — only the scoring was broken
for AQuA-RAT (MC answers not detected from HF loader) and AI2D (string answer
not converted to int). This script re-reads every CSV and recomputes
statistics.json with the fixed parser.

CPU only, takes seconds. No GPU needed.

Usage:
    python scripts/rescore_phase3.py                          # rescore all
    python scripts/rescore_phase3.py --models InternVL2-8B    # specific model
    python scripts/rescore_phase3.py --benchmarks aqua_rat,ai2d  # specific benchmarks
    python scripts/rescore_phase3.py --dry-run                # show what would change
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.evaluation import (
    compute_accuracy, mcnemar_test, bootstrap_ci,
    binomial_ci, cohens_h, compute_all_statistics,
    format_statistics_report, score_mismatch_follows,
    extract_numeric_answer,
)
from src.benchmark_eval import match_answer, classify_benchmark_error
from src.benchmarks import BenchmarkItem, BENCHMARK_REGISTRY, extract_number_from_answer


def rescore_answer(prediction, reference, benchmark_name):
    """Re-match a prediction against its reference using fixed logic."""
    ref_str = str(reference).strip()

    # Multiple-choice: reference is a single letter A-E (AQuA-RAT)
    if len(ref_str) == 1 and ref_str.upper() in "ABCDE":
        correct_letter = ref_str.upper()
        pred_upper = str(prediction).upper()
        letters = re.findall(r'\b([A-E])\b', pred_upper)
        if letters:
            return letters[-1] == correct_letter
        canon = re.search(r"####\s*([A-E])", pred_upper)
        if canon:
            return canon.group(1) == correct_letter
        return False

    # Multiple-choice by index: only for benchmarks that use integer indices (AI2D, ScienceQA)
    mc_index_benchmarks = {"ai2d", "scienceqa"}
    if benchmark_name in mc_index_benchmarks:
        try:
            ref_int = int(float(reference))
            if 0 <= ref_int <= 4:
                correct_letter = chr(65 + ref_int)
                pred_upper = str(prediction).upper()
                letters = re.findall(r'\b([A-E])\b', pred_upper)
                if letters:
                    return letters[-1] == correct_letter
                return False
        except (ValueError, TypeError):
            pass

    # Numeric comparison
    import math
    pred_num = extract_numeric_answer(str(prediction))
    ref_num = extract_numeric_answer(ref_str)
    if pred_num is not None and ref_num is not None:
        if math.isinf(pred_num) or math.isnan(pred_num):
            return False
        if math.isinf(ref_num) or math.isnan(ref_num):
            return False
        return round(pred_num) == round(ref_num)
    return False


def rescore_protocol_a(csv_path, benchmark_name, model_name):
    """Rescore a Protocol A (text-rendered) benchmark CSV."""
    df = pd.read_csv(csv_path)

    text_correct = []
    img_correct = []
    mm_follows = []

    for _, row in df.iterrows():
        text_correct.append(rescore_answer(row['pred_text'], row['reference'], benchmark_name))
        img_correct.append(rescore_answer(row['pred_rendered'], row['reference'], benchmark_name))
        mm_follows.append(row['mismatch_follows'])

    stats = compute_all_statistics(text_correct, img_correct, mm_follows)
    stats['benchmark'] = benchmark_name
    stats['model'] = model_name
    return stats


def rescore_protocol_b(csv_path, benchmark_name, model_name):
    """Rescore a Protocol B (native visual) benchmark CSV."""
    df = pd.read_csv(csv_path)

    mm_correct = []
    text_correct = []

    for _, row in df.iterrows():
        mm_correct.append(rescore_answer(row['pred_multimodal'], row['reference'], benchmark_name))
        text_correct.append(rescore_answer(row['pred_text'], row['reference'], benchmark_name))

    n = len(mm_correct)
    acc_mm = compute_accuracy(mm_correct)
    acc_text = compute_accuracy(text_correct)
    mm_ci = binomial_ci(sum(mm_correct), n)
    text_ci = binomial_ci(sum(text_correct), n)
    mm_boot = bootstrap_ci(mm_correct)
    text_boot = bootstrap_ci(text_correct)
    mcn_chi2, mcn_p, mcn_b, mcn_c = mcnemar_test(mm_correct, text_correct)
    effect = cohens_h(acc_mm, acc_text)

    stats = {
        "n": n,
        "benchmark": benchmark_name,
        "model": model_name,
        "acc_multimodal": acc_mm,
        "acc_text_only": acc_text,
        "acc_diff": acc_mm - acc_text,
        "multimodal_ci_95": mm_ci,
        "text_ci_95": text_ci,
        "multimodal_boot_ci_95": mm_boot,
        "text_boot_ci_95": text_boot,
        "mcnemar_chi2": mcn_chi2,
        "mcnemar_p": mcn_p,
        "mcnemar_b": mcn_b,
        "mcnemar_c": mcn_c,
        "cohens_h": effect,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Rescore Phase 3 results from CSVs")
    parser.add_argument("--results-dir", default=None,
                        help="Phase 3 results directory")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated benchmark names")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args()

    # Find results directory
    results_dir = args.results_dir
    if not results_dir:
        for candidate in [
            os.path.expanduser("~/vlm_research_results/phase3"),
            "results/phase3",
        ]:
            if os.path.isdir(candidate):
                results_dir = candidate
                break

    if not results_dir or not os.path.isdir(results_dir):
        print(f"Results directory not found. Use --results-dir")
        sys.exit(1)

    print(f"Results directory: {results_dir}")

    # Determine which benchmarks use which protocol
    protocol_a = {"gsm8k", "svamp", "aqua_rat", "math"}
    protocol_b = {"mathvista", "scienceqa", "ai2d", "chartqa"}

    filter_models = set(args.models.split(",")) if args.models else None
    filter_benchmarks = set(args.benchmarks.split(",")) if args.benchmarks else None

    changes = []

    for model_dir in sorted(os.listdir(results_dir)):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        if filter_models and model_dir not in filter_models:
            continue

        for bench_dir in sorted(os.listdir(model_path)):
            bench_path = os.path.join(model_path, bench_dir)
            if not os.path.isdir(bench_path):
                continue
            if filter_benchmarks and bench_dir not in filter_benchmarks:
                continue

            csv_path = os.path.join(bench_path, "results.csv")
            stats_path = os.path.join(bench_path, "statistics.json")

            if not os.path.exists(csv_path):
                continue

            # Load old stats for comparison
            old_stats = {}
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    old_stats = json.load(f)

            # Determine model full name from old stats
            model_name = old_stats.get("model", model_dir)

            # Rescore
            if bench_dir in protocol_a:
                new_stats = rescore_protocol_a(csv_path, bench_dir, model_name)
            elif bench_dir in protocol_b:
                new_stats = rescore_protocol_b(csv_path, bench_dir, model_name)
            else:
                print(f"  SKIP {model_dir}/{bench_dir} — unknown protocol")
                continue

            # Compare
            if bench_dir in protocol_a:
                old_text = old_stats.get("acc_text", -1)
                old_img = old_stats.get("acc_img", -1)
                new_text = new_stats.get("acc_text", -1)
                new_img = new_stats.get("acc_img", -1)
                changed = abs(old_text - new_text) > 0.001 or abs(old_img - new_img) > 0.001
                label = f"Text: {old_text:.1%}->{new_text:.1%}  Image: {old_img:.1%}->{new_img:.1%}"
            else:
                old_mm = old_stats.get("acc_multimodal", -1)
                old_t = old_stats.get("acc_text_only", -1)
                new_mm = new_stats.get("acc_multimodal", -1)
                new_t = new_stats.get("acc_text_only", -1)
                changed = abs(old_mm - new_mm) > 0.001 or abs(old_t - new_t) > 0.001
                label = f"MM: {old_mm:.1%}->{new_mm:.1%}  Text: {old_t:.1%}->{new_t:.1%}"

            status = "CHANGED" if changed else "OK"
            print(f"  {model_dir}/{bench_dir}: {status}  {label}")

            if changed:
                changes.append((bench_path, new_stats))

    print(f"\n{'='*60}")
    print(f"  {len(changes)} benchmark(s) need rescoring")
    print(f"{'='*60}")

    if args.dry_run:
        print("  (dry run — no files written)")
        return

    for bench_path, new_stats in changes:
        stats_path = os.path.join(bench_path, "statistics.json")
        serializable = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in new_stats.items()}
        with open(stats_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Updated: {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
