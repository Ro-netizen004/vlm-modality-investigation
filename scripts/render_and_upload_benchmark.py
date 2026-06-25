#!/usr/bin/env python3
"""
Render text-based benchmark questions as images and upload to HuggingFace.

Same approach as GSM8K v2 dataset: render once, upload once, reuse everywhere.

Usage:
    python scripts/render_and_upload_benchmark.py --benchmark svamp
    python scripts/render_and_upload_benchmark.py --benchmark svamp --render-only
    python scripts/render_and_upload_benchmark.py --benchmark svamp --upload-only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset, Features, Value, Image as HFImage
from PIL import Image
from tqdm import tqdm

from src.benchmarks import load_benchmark, BENCHMARK_REGISTRY
from src.rendering import render_text_to_image


def render_benchmark_images(benchmark_name: str, output_dir: str):
    """Render all questions in a benchmark as PNG images."""
    items = load_benchmark(benchmark_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRendering {len(items)} {benchmark_name} questions as images...")
    for item in tqdm(items, desc=f"Rendering {benchmark_name}"):
        path = os.path.join(output_dir, f"q{item.id:04d}.png")
        if not os.path.exists(path):
            img = render_text_to_image(item.question, width=900, font_size=22, padding=40)
            img.save(path)

    print(f"Done — {len(items)} images saved to {output_dir}")
    return items


def upload_to_hf(benchmark_name: str, image_dir: str, items, repo_id: str):
    """Upload rendered images + questions + answers as a HuggingFace dataset."""
    from huggingface_hub import HfApi

    records = []
    for item in tqdm(items, desc="Building dataset"):
        img_path = os.path.join(image_dir, f"q{item.id:04d}.png")
        records.append({
            "problem_id": item.id,
            "question": item.question,
            "answer": item.reference_answer,
            "image": img_path,
        })

    ds = Dataset.from_dict({
        "problem_id": [r["problem_id"] for r in records],
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "image": [r["image"] for r in records],
    })
    ds = ds.cast_column("image", HFImage())

    print(f"\nUploading to HuggingFace: {repo_id}")
    ds.push_to_hub(repo_id, split="train")
    print(f"Done — dataset uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Render and upload benchmark images to HF")
    parser.add_argument("--benchmark", required=True,
                        choices=["svamp", "aqua_rat", "math"],
                        help="Which benchmark to render")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save rendered images")
    parser.add_argument("--render-only", action="store_true",
                        help="Only render, don't upload")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload (images must exist)")
    args = parser.parse_args()

    info = BENCHMARK_REGISTRY[args.benchmark]
    repo_id = info["hf_repo_id"]
    if not repo_id:
        print(f"No HF repo ID configured for {args.benchmark}")
        sys.exit(1)

    output_dir = args.output_dir or f"rendered_images/{args.benchmark}"

    if args.upload_only:
        items = load_benchmark(args.benchmark)
        upload_to_hf(args.benchmark, output_dir, items, repo_id)
    elif args.render_only:
        render_benchmark_images(args.benchmark, output_dir)
    else:
        items = render_benchmark_images(args.benchmark, output_dir)
        upload_to_hf(args.benchmark, output_dir, items, repo_id)


if __name__ == "__main__":
    main()
