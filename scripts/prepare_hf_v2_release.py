#!/usr/bin/env python3
"""
Build a Hugging Face dataset folder for GSM8K rendered images v2 (Phase 1 protocol).

Usage:
    python scripts/prepare_hf_v2_release.py \\
        --images-dir "D:/Downloads/rendered_images" \\
        --output-dir "./hf_gsm8k_v2_upload"

Then upload output-dir to a new HF dataset repo (see docs/HF_V2_UPLOAD.md).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

RENDER_CONFIG_V2 = {
    "version": "v2",
    "source": "openai/gsm8k",
    "config": "main",
    "split": "test",
    "renderer": "src/rendering.py::render_text_to_image",
    "width_px": 900,
    "font_size_px": 22,
    "padding_px": 40,
    "bg_color": "white",
    "text_color": "black",
    "text_prefix": None,
    "wrap_method": "textwrap.fill",
    "naming": "q{id:03d}.png",
    "parent_study": "vlm-modality-research Phase 1",
    "notes": "Distinct from v1 (RodelaG/gsm8k-rendered-vlm): 672px, Solve-this-step-by-step prefix, q0000.png (4-digit) naming.",
}


def extract_answer(text: str) -> str:
    match = re.findall(r"####\s*([+-]?\d+(?:\.\d+)?)", str(text))
    return match[-1] if match else str(text).strip()


def extract_reasoning(text: str) -> str:
    reasoning = str(text).split("####")[0]
    reasoning = re.sub(r"<<.*?>>", "", reasoning)
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r"\n\s*\n+", "\n", reasoning)
    return reasoning.strip()


def detect_image_name(i: int, images_dir: Path) -> str | None:
    for fmt in (f"q{i:03d}.png", f"q{i:04d}.png"):
        if (images_dir / fmt).exists():
            return fmt
    return None


def main():
    p = argparse.ArgumentParser(description="Prepare HF v2 dataset upload folder")
    p.add_argument(
        "--images-dir",
        required=True,
        help="Folder with PNGs from Drive vlm_research_results/rendered_images",
    )
    p.add_argument(
        "--output-dir",
        default="hf_gsm8k_v2_upload",
        help="Folder to create with rendered_images/ + data/",
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy PNGs into output-dir/rendered_images (default: symlink on Unix, copy on Windows)",
    )
    args = p.parse_args()

    images_dir = Path(args.images_dir).resolve()
    out = Path(args.output_dir).resolve()
    out_images = out / "rendered_images"
    out_data = out / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")

    print("Loading openai/gsm8k test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = len(ds)
    print(f"GSM8K test: {n} problems")

    rows = []
    missing = []
    naming = None
    for i in tqdm(range(n), desc="Validate images"):
        fname = detect_image_name(i, images_dir)
        if fname is None:
            missing.append(i)
            continue
        if naming is None:
            # q000.png (len 8) vs q0000.png (len 9) vs q1000.png (len 9)
            naming = "03d" if len(fname) <= 8 else "04d"
        rel = f"rendered_images/{fname}"
        raw_ans = ds[i]["answer"]
        rows.append(
            {
                "id": i,
                "question": ds[i]["question"],
                "answer": extract_answer(raw_ans),
                "image": rel,
                "reasoning": extract_reasoning(raw_ans),
            }
        )

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images (e.g. id {missing[:5]}). "
            f"Expected qXXX.png in {images_dir}"
        )

    print(f"All {n} images found ({naming} naming)")

    # Metadata + config
    meta_path = out_data / "gsm8k_metadata_clean.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    cfg = {
        **RENDER_CONFIG_V2,
        "naming": "q{id:03d}.png (id 0-999); q{id}.png (id 1000+)",
        "naming_detected": naming,
        "num_images": n,
    }
    with open(out_data / "render_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Images into bundle
    if out_images.exists():
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True)

    if args.copy_images or not hasattr(Path, "symlink_to"):
        print("Copying images...")
        for row in tqdm(rows, desc="Copy"):
            src = images_dir / Path(row["image"]).name
            shutil.copy2(src, out_images / src.name)
    else:
        print("Symlinking images...")
        for row in rows:
            src = images_dir / Path(row["image"]).name
            (out_images / src.name).symlink_to(src)

    print(f"\nReady to upload: {out}")
    print(f"  {out_images}/  ({n} PNGs)")
    print(f"  {meta_path}")
    print(f"  {out_data / 'render_config.json'}")
    print("\nNext: docs/HF_V2_UPLOAD.md")


if __name__ == "__main__":
    main()
