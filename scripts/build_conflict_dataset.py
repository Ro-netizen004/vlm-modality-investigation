#!/usr/bin/env python3
"""
Build the VLM modality-arbitration conflict dataset (v1).

Consolidates the controlled text-image *conflict* stimuli used in the Phase 6
legibility experiment into a single standardized, model-agnostic dataset that
others can run their own VLM against.

One row == one conflict instance:

    image  = rendered problem  i           (degraded at `degradation_level`)
    text   = problem (i+1) % n              (the conflicting statement)
    the two ground-truth answers are carried side by side, so "which modality
    did the model follow?" is decidable.

This is the MISMATCH construction from scripts/run_legibility.py, made
reproducible outside any model run:
    * text fields + both answers come from the (model-agnostic) legibility CSVs,
      which are exactly what was presented/scored;
    * image bytes are regenerated from the canonical HF renders with the same
      noise transform and seed (src.noise.apply_noise_level, seed = 42 + i),
      so Level 0 is pixel-identical to the Phase 1/3 baseline.

Schema is `channel`-keyed so the text-degradation mirror arm (Phase 7) drops in
later as rows with channel="text" — no schema change, no rework.

Usage:
    # fast validation — assemble metadata only, no image generation
    python scripts/build_conflict_dataset.py --dry-run

    # small end-to-end smoke test (10 problems, levels 0 & 5, svamp only)
    python scripts/build_conflict_dataset.py --sources svamp --levels 0 5 --limit 10 \
        --out data/conflict_dataset_smoke

    # full v1 build (gsm8k + svamp, image arm, levels 0/2/4/5)
    python scripts/build_conflict_dataset.py --out data/conflict_dataset_v1
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.noise import NOISE_LEVELS, apply_noise_level  # noqa: E402

# ── Source config ──────────────────────────────────────────────────────────────
# Stimuli are identical across models (verified); use one fully-covered model as
# the canonical CSV source for the model-agnostic text fields + answers.
REPO_ROOT = Path(__file__).resolve().parent.parent
LEGIBILITY_ROOT = REPO_ROOT / "results" / "phase6_legibility"
STIMULUS_MODEL = "InternVL2-8B"  # has full gsm8k + svamp coverage, levels 0/2/4/5

SOURCES = {
    "gsm8k": {
        "hf_repo": "vlm-modality-research/gsm8k-rendered-vlm-v2",
        "n": 1319,
    },
    "svamp": {
        "hf_repo": "vlm-modality-research/svamp-rendered-vlm-v1",
        "n": 300,
    },
}

# Image-degradation ladder used by Phase 6 (monotonic clean -> heavy).
DEFAULT_LEVELS = [0, 2, 4, 5]
IMAGE_SEED_BASE = 42  # run_legibility uses seed = IMAGE_SEED_BASE + i per image

# The exact prompt wrapper the text side was presented with (for reproducibility).
def mismatch_prompt(text_question: str) -> str:
    return ("Solve the following math problem step by step. "
            "End with '#### <answer>'.\n\n"
            f"Problem: {text_question}")


def load_stimuli(source: str) -> pd.DataFrame:
    """Model-agnostic conflict stimuli for one source, read from the level-0 CSV.

    Text pairing + answers are level-independent (only the image degrades), so a
    single CSV defines all the non-image columns. Returns a frame indexed 0..n-1
    by problem_id with columns: image_question, text_question, image_answer,
    text_answer.
    """
    csv_path = LEGIBILITY_ROOT / STIMULUS_MODEL / source / "level_0_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing stimulus CSV: {csv_path}")
    df = pd.read_csv(csv_path)  # proper CSV parse — fields contain commas/newlines
    df = df.sort_values("problem_id").reset_index(drop=True)
    keep = df[["problem_id", "image_question", "text_question",
               "image_reference", "text_reference"]].copy()
    keep = keep.rename(columns={"image_reference": "image_answer",
                                "text_reference": "text_answer"})
    return keep


def load_canonical_images(source: str, expected_questions):
    """Return list[PIL.Image] of the CLEAN canonical renders, index-aligned to
    problem_id, after asserting the HF question text matches the CSV stimuli."""
    from datasets import load_dataset
    repo = SOURCES[source]["hf_repo"]
    ds = load_dataset(repo)
    split = list(ds.keys())[0]
    d = ds[split].sort("problem_id")
    if len(d) != len(expected_questions):
        raise ValueError(f"{source}: HF n={len(d)} != CSV n={len(expected_questions)}")
    images, mismatches = [], 0
    for i in range(len(d)):
        row = d[i]
        # Belt-and-suspenders: the image we attach must be the same problem the
        # CSV labeled as image_question[i]. Compare on a normalized prefix
        # (rendering strips nothing, but encodings of curly quotes can differ).
        hf_q = str(row["question"]).strip()
        csv_q = str(expected_questions[i]).strip()
        if hf_q[:40] != csv_q[:40]:
            mismatches += 1
        img = row["image"]
        images.append(img.convert("RGB") if img.mode != "RGB" else img)
    if mismatches:
        raise ValueError(f"{source}: {mismatches} HF/CSV question mismatches — "
                         f"ordering is not aligned, aborting")
    return images


def build_rows(source, levels, limit, with_images):
    stim = load_stimuli(source)
    n = len(stim)
    if limit:
        n = min(n, limit)
    clean_images = None
    if with_images:
        clean_images = load_canonical_images(source, stim["image_question"].tolist())

    rows = []
    for level in levels:
        name = NOISE_LEVELS[level]["name"]
        for i in range(n):
            rec = {
                "id": f"{source}-image-L{level}-{i:04d}",
                "source": source,
                "conflict_type": "mismatch",
                "channel": "image",
                "degradation_level": level,
                "degradation_name": name,
                "image_problem_id": i,
                "text_problem_id": (i + 1) % len(stim),
                "image_question": stim.at[i, "image_question"],
                "text_question": stim.at[i, "text_question"],
                "image_answer": str(stim.at[i, "image_answer"]),
                "text_answer": str(stim.at[i, "text_answer"]),
                "prompt": mismatch_prompt(stim.at[i, "text_question"]),
                "image_seed": IMAGE_SEED_BASE + i,
            }
            if with_images:
                rec["image"] = apply_noise_level(
                    clean_images[i], level,
                    text=stim.at[i, "image_question"], seed=IMAGE_SEED_BASE + i)
            rows.append(rec)
    return rows, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES),
                    choices=list(SOURCES))
    ap.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS,
                    choices=list(NOISE_LEVELS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap problems per source (smoke tests)")
    ap.add_argument("--out", default="data/conflict_dataset_v1")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble metadata only; skip image generation and save")
    args = ap.parse_args()

    with_images = not args.dry_run
    all_rows = []
    for source in args.sources:
        rows, n = build_rows(source, args.levels, args.limit, with_images)
        print(f"{source:8s} n={n:4d}  levels={args.levels}  -> {len(rows):5d} rows")
        all_rows.append((source, rows))

    total = sum(len(r) for _, r in all_rows)
    print(f"\nTOTAL rows: {total}")

    # Show one sample (without the image object) so the schema is legible.
    sample = {k: v for k, v in all_rows[0][1][0].items() if k != "image"}
    import json
    print("\nSample row (image omitted):")
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:1200])

    if args.dry_run:
        print("\n[dry-run] metadata assembled OK; no images generated, nothing written.")
        return

    from datasets import Dataset, Features, Value, Image as HFImage
    features = Features({
        "id": Value("string"),
        "source": Value("string"),
        "conflict_type": Value("string"),
        "channel": Value("string"),
        "degradation_level": Value("int32"),
        "degradation_name": Value("string"),
        "image_problem_id": Value("int32"),
        "text_problem_id": Value("int32"),
        "image": HFImage(),
        "image_question": Value("string"),
        "text_question": Value("string"),
        "image_answer": Value("string"),
        "text_answer": Value("string"),
        "prompt": Value("string"),
        "image_seed": Value("int32"),
    })
    flat = [r for _, rows in all_rows for r in rows]
    ds = Dataset.from_list(flat, features=features)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"\nSaved dataset ({len(ds)} rows) -> {out}")
    print("Columns:", ds.column_names)


if __name__ == "__main__":
    main()
