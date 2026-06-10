#!/usr/bin/env python3
"""
Multi-benchmark runner — evaluates VLMs across all 8 benchmarks.

Usage:
    python scripts/run_multi_benchmark.py                                     # all
    python scripts/run_multi_benchmark.py --benchmarks gsm8k,svamp            # specific
    python scripts/run_multi_benchmark.py --models "Qwen/Qwen2-VL-2B-Instruct"
    python scripts/run_multi_benchmark.py --num-problems 50                   # quick test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks import load_benchmark, get_benchmark_info, list_benchmarks, BENCHMARK_REGISTRY
from src.benchmark_eval import (
    run_protocol_a, run_protocol_b,
    save_protocol_a_results, save_protocol_b_results,
)
from src.models import VLMModel


def main():
    parser = argparse.ArgumentParser(description="Multi-Benchmark VLM Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated benchmark names (default: all)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (overrides config)")
    parser.add_argument("--num-problems", type=int, default=None,
                        help="Limit problems per benchmark")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config.get("seed", 42))
    output_dir = args.output_dir or config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Select benchmarks
    if args.benchmarks:
        benchmark_names = [b.strip() for b in args.benchmarks.split(",")]
    else:
        benchmark_names = list(BENCHMARK_REGISTRY.keys())

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

    all_results = {}

    for mc in model_configs:
        print(f"\n{'#' * 70}")
        print(f"  MODEL: {mc['name']}")
        print(f"{'#' * 70}\n")

        vlm = VLMModel(
            model_name=mc["name"], model_type=mc["type"],
            max_new_tokens=mc.get("max_new_tokens", 256),
            torch_dtype=mc.get("torch_dtype", "bfloat16"),
            quantize=mc.get("quantize"),
        )
        vlm.load()

        model_results = {}

        for bench_name in benchmark_names:
            info = get_benchmark_info(bench_name)
            short = mc["name"].split("/")[-1]

            # Skip if already done
            stats_path = os.path.join(output_dir, short, bench_name, "statistics.json")
            if os.path.exists(stats_path):
                print(f"\n>>> SKIP {bench_name} for {short} — already done <<<\n")
                with open(stats_path) as f:
                    model_results[bench_name] = json.load(f)
                continue

            print(f"\n{'=' * 60}")
            print(f"  BENCHMARK: {bench_name}")
            print(f"{'=' * 60}")

            t0 = time.time()
            items = load_benchmark(bench_name, args.num_problems)

            if not info["has_images"]:
                # Protocol A: text-based, render as images
                image_dir = os.path.join(output_dir, "images", bench_name)
                results = run_protocol_a(vlm, items, bench_name, output_dir, image_dir)
                stats = save_protocol_a_results(
                    results, items, mc["name"], bench_name, output_dir)
            else:
                # Protocol B: visual benchmarks with native images
                results, eval_items = run_protocol_b(vlm, items, bench_name, output_dir)
                stats = save_protocol_b_results(
                    results, eval_items, mc["name"], bench_name, output_dir)

            elapsed = time.time() - t0
            stats["elapsed_minutes"] = round(elapsed / 60, 1)
            model_results[bench_name] = stats

        all_results[mc["name"]] = model_results
        vlm.unload()

    # Cross-benchmark summary
    print("\n\n" + "=" * 80)
    print("  CROSS-BENCHMARK SUMMARY")
    print("=" * 80)
    for model, benchmarks in all_results.items():
        short = model.split("/")[-1]
        print(f"\n  {short}:")
        for bench, stats in benchmarks.items():
            if "acc_text" in stats:
                print(f"    {bench:12s}  Text={stats['acc_text']:.3f}  "
                      f"Image={stats['acc_img']:.3f}  "
                      f"Drop={stats['acc_drop']:+.3f}  "
                      f"p={stats['mcnemar_p']:.4f}")
            elif "acc_multimodal" in stats:
                print(f"    {bench:12s}  MM={stats['acc_multimodal']:.3f}  "
                      f"Text={stats['acc_text_only']:.3f}  "
                      f"Diff={stats['acc_diff']:+.3f}  "
                      f"p={stats['mcnemar_p']:.4f}")

    summary_path = os.path.join(output_dir, "cross_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
