#!/usr/bin/env python3
"""
Render and upload text-based benchmark datasets to HuggingFace.

Handles: GSM8K (update existing), SVAMP, AQuA-RAT, MATH (create new).
Each dataset is uploaded with: image, problem_id, question, answer, split.

Usage:
    python scripts/prepare_hf_datasets.py --datasets gsm8k,svamp
    python scripts/prepare_hf_datasets.py --datasets all
    python scripts/prepare_hf_datasets.py --datasets gsm8k --dry-run
    python scripts/prepare_hf_datasets.py --datasets svamp --num-problems 50  # test run
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rendering import render_text_to_image

HF_ORG = "vlm-modality-research"

DATASET_CONFIGS = {
    "gsm8k": {
        "repo_id": f"{HF_ORG}/gsm8k-rendered-vlm-v2",
        "description": "GSM8K test split with rendered images (v2: 900px, DejaVu Sans 22pt)",
    },
    "svamp": {
        "repo_id": f"{HF_ORG}/svamp-rendered-vlm-v1",
        "description": "SVAMP test split with rendered images (900px, DejaVu Sans 22pt)",
    },
    "aqua_rat": {
        "repo_id": f"{HF_ORG}/aqua-rat-rendered-vlm-v1",
        "description": "AQuA-RAT test split with rendered images (900px, DejaVu Sans 22pt)",
    },
    "math": {
        "repo_id": f"{HF_ORG}/math-rendered-vlm-v1",
        "description": "MATH competition test split with rendered images (900px, DejaVu Sans 22pt)",
    },
}


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_gsm8k_raw(num_problems=None):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))
    rows = []
    for i, row in enumerate(ds):
        # Extract just the numeric answer from #### format
        match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        answer = match.group(1).replace(",", "") if match else row["answer"].strip()
        rows.append({
            "problem_id": i,
            "question": row["question"],
            "answer": answer,
            "split": "test",
        })
    print(f"GSM8K: {len(rows)} problems loaded")
    return rows


def load_svamp_raw(num_problems=None):
    ds = load_dataset("ChilleD/SVAMP", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))
    rows = []
    for i, row in enumerate(ds):
        question = f"{row['Body']} {row['Question']}".strip()
        rows.append({
            "problem_id": i,
            "question": question,
            "answer": str(row["Answer"]),
            "split": "test",
        })
    print(f"SVAMP: {len(rows)} problems loaded")
    return rows


def load_aqua_rat_raw(num_problems=None):
    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))
    rows = []
    for i, row in enumerate(ds):
        options = row["options"]
        choices_text = "\n".join(options)
        question = f"{row['question']}\n\nOptions:\n{choices_text}"
        rows.append({
            "problem_id": i,
            "question": question,
            "answer": row["correct"],
            "split": "test",
        })
    print(f"AQuA-RAT: {len(rows)} problems loaded")
    return rows


def load_math_raw(num_problems=None):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))
    rows = []
    for i, row in enumerate(ds):
        # Extract boxed answer
        solution = row.get("solution", row.get("answer", ""))
        boxed = re.search(r"\\boxed\{([^}]+)\}", solution)
        answer = boxed.group(1) if boxed else solution.strip()
        question = row.get("problem", row.get("question", ""))
        rows.append({
            "problem_id": i,
            "question": question,
            "answer": answer,
            "split": "test",
        })
    print(f"MATH: {len(rows)} problems loaded")
    return rows


LOADERS = {
    "gsm8k":   load_gsm8k_raw,
    "svamp":   load_svamp_raw,
    "aqua_rat": load_aqua_rat_raw,
    "math":    load_math_raw,
}


# ── Render + build HF dataset ─────────────────────────────────────────────────

def build_dataset(name: str, num_problems: int = None) -> Dataset:
    """Load raw data, render images, return HF Dataset."""
    rows = LOADERS[name](num_problems)

    print(f"Rendering {len(rows)} images for {name}...")
    images = []
    for row in tqdm(rows, desc="Rendering"):
        img = render_text_to_image(row["question"])
        images.append(img)

    dataset = Dataset.from_dict({
        "problem_id": [r["problem_id"] for r in rows],
        "question":   [r["question"]   for r in rows],
        "answer":     [r["answer"]     for r in rows],
        "split":      [r["split"]      for r in rows],
        "image":      images,
    })
    return dataset


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Render and upload benchmark datasets to HF")
    parser.add_argument("--datasets", default="all",
                        help="Comma-separated: gsm8k,svamp,aqua_rat,math or 'all'")
    parser.add_argument("--num-problems", type=int, default=None,
                        help="Limit problems (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render and build dataset but don't push to HF")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HuggingFace token required. Pass --hf-token or set HF_TOKEN env var.")
        sys.exit(1)

    if args.datasets == "all":
        names = list(DATASET_CONFIGS.keys())
    else:
        names = [n.strip() for n in args.datasets.split(",")]

    for name in names:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}. Choose from: {list(DATASET_CONFIGS.keys())}")
            continue

        cfg = DATASET_CONFIGS[name]
        print(f"\n{'=' * 60}")
        print(f"  {name.upper()} → {cfg['repo_id']}")
        print(f"{'=' * 60}")

        dataset = build_dataset(name, args.num_problems)
        print(f"Built dataset: {len(dataset)} rows, columns: {dataset.column_names}")

        if args.dry_run:
            print(f"[DRY RUN] Would push to {cfg['repo_id']}")
            print(f"  Sample row 0: problem_id={dataset[0]['problem_id']}, "
                  f"question={dataset[0]['question'][:60]}..., "
                  f"answer={dataset[0]['answer']}, "
                  f"image_size={dataset[0]['image'].size}")
            continue

        print(f"Pushing to {cfg['repo_id']}...")
        dataset.push_to_hub(
            cfg["repo_id"],
            token=token,
            commit_message=f"Add {name} rendered dataset with question/answer/image",
        )
        print(f"Done: huggingface.co/datasets/{cfg['repo_id']}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
