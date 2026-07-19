#!/usr/bin/env python3
"""
Build the VLM modality-arbitration conflict dataset (v1).

Consolidates the controlled text-image *conflict* stimuli used in the Phase 6
legibility experiment into a single standardized, model-agnostic dataset that
others can run their own VLM against.

One row == one conflict instance. Two symmetric arms share one schema:

    channel="image" (Phase 6): image = rendered problem i, DEGRADED at `level`;
                               text  = clean problem (i+1)          (conflict via image)
    channel="text"  (Phase 7): image = rendered problem i, CLEAN (level 0);
                               text  = problem (i+1), DEGRADED at `level`  (conflict via text)

In both, the two ground-truth answers are carried side by side, so "which
modality did the model follow?" is decidable from output text alone.

This is the MISMATCH construction from scripts/run_legibility.py, made
reproducible outside any model run:
    * text fields + both answers come from the (model-agnostic) legibility CSVs,
      which are exactly what was presented/scored;
    * IMAGE arm: image bytes are regenerated from the canonical HF renders with the
      same noise transform and seed (src.noise.apply_noise_level, seed = 42 + i),
      so Level 0 is pixel-identical to the Phase 1/3 baseline;
    * TEXT arm: the image is the clean canonical render; the conflicting text is
      corrupted with src.text_noise.degrade_text(text, level, seed=text_problem_id),
      matching run_legibility's --channel text seeding exactly.

Column convention (consistent across arms): the *_question columns are always the
CLEAN problem text (labels), and the degradation always lives in the presented
modality — the `image` bytes (image arm) or the `prompt` string (text arm). So
`prompt` always contains the real, possibly-corrupted text the model was shown,
and is reproducible from (text_question, degradation_level, text_problem_id).

Schema is `channel`-keyed, so both arms coexist with no schema change.

Usage:
    # fast validation — assemble metadata only, no image generation
    python scripts/build_conflict_dataset.py --dry-run

    # small end-to-end smoke test (10 problems, levels 0 & 5, svamp, both arms)
    python scripts/build_conflict_dataset.py --sources svamp --levels 0 5 --limit 10 \
        --out data/conflict_dataset_smoke

    # full v2 build — BOTH arms (gsm8k + svamp, image+text, levels 0/2/4/5)
    python scripts/build_conflict_dataset.py --out data/conflict_dataset_v2

    # single-arm builds
    python scripts/build_conflict_dataset.py --channel image --out data/conflict_dataset_v1
    python scripts/build_conflict_dataset.py --channel text  --out data/conflict_dataset_text
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.noise import NOISE_LEVELS, apply_noise_level  # noqa: E402
from src.text_noise import TEXT_NOISE_LEVELS, degrade_text  # noqa: E402


def ladder_for(channel):
    """Degradation ladder + level-name map for a channel (image vs text noise)."""
    return TEXT_NOISE_LEVELS if channel == "text" else NOISE_LEVELS

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


def build_rows(source, levels, limit, with_images, channels):
    stim = load_stimuli(source)
    n_full = len(stim)
    n = min(n_full, limit) if limit else n_full
    clean_images = None
    if with_images:
        clean_images = load_canonical_images(source, stim["image_question"].tolist())

    rows = []
    for channel in channels:
        ladder = ladder_for(channel)
        chan_levels = [L for L in levels if L in ladder]
        skipped = [L for L in levels if L not in ladder]
        if skipped:
            print(f"  [{source}/{channel}] skipping levels {skipped} "
                  f"(supported: {sorted(ladder)})")
        for level in chan_levels:
            name = ladder[level]["name"]
            for i in range(n):
                text_pid = (i + 1) % n_full  # conflicting text = problem i+1
                # The *_question columns stay CLEAN in both arms; the degradation
                # lives in the presented modality (image bytes / prompt string).
                if channel == "text":
                    # Mirror arm: image clean, text corrupted (seed matches
                    # run_legibility --channel text, seed = text_problem_id).
                    presented_text = degrade_text(
                        stim.at[i, "text_question"], level, seed=text_pid)
                    image = clean_images[i] if with_images else None
                else:
                    # Image arm: text clean, image degraded at this level.
                    presented_text = stim.at[i, "text_question"]
                    image = (apply_noise_level(
                        clean_images[i], level,
                        text=stim.at[i, "image_question"], seed=IMAGE_SEED_BASE + i)
                        if with_images else None)
                rec = {
                    "id": f"{source}-{channel}-L{level}-{i:04d}",
                    "source": source,
                    "conflict_type": "mismatch",
                    "channel": channel,
                    "degradation_level": level,
                    "degradation_name": name,
                    "image_problem_id": i,
                    "text_problem_id": text_pid,
                    "image_question": stim.at[i, "image_question"],
                    "text_question": stim.at[i, "text_question"],
                    "image_answer": str(stim.at[i, "image_answer"]),
                    "text_answer": str(stim.at[i, "text_answer"]),
                    "prompt": mismatch_prompt(presented_text),
                    "image_seed": IMAGE_SEED_BASE + i,
                }
                if with_images:
                    rec["image"] = image
                rows.append(rec)
    return rows, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES),
                    choices=list(SOURCES))
    ap.add_argument("--channel", default="both", choices=["image", "text", "both"],
                    help="which conflict arm(s) to build: 'image' (Phase 6), "
                         "'text' (Phase 7 mirror), or 'both' (default). Levels not "
                         "in a channel's ladder are skipped for that channel; the "
                         "text ladder supports 0/2/4/5.")
    ap.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS,
                    choices=list(NOISE_LEVELS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap problems per source (smoke tests)")
    ap.add_argument("--out", default="data/conflict_dataset_v2")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble metadata only; skip image generation and save")
    args = ap.parse_args()

    channels = ["image", "text"] if args.channel == "both" else [args.channel]

    with_images = not args.dry_run
    all_rows = []
    for source in args.sources:
        rows, n = build_rows(source, args.levels, args.limit, with_images, channels)
        print(f"{source:8s} n={n:4d}  channels={channels}  levels={args.levels}  "
              f"-> {len(rows):5d} rows")
        all_rows.append((source, rows))

    flat = [r for _, rows in all_rows for r in rows]
    total = len(flat)
    from collections import Counter
    by_chan = Counter(r["channel"] for r in flat)
    print(f"\nTOTAL rows: {total}  ({dict(by_chan)})")

    # Show one sample per channel (without the image object) so the schema is legible.
    import json
    for chan in channels:
        ex = next((r for r in flat if r["channel"] == chan), None)
        if ex is None:
            continue
        sample = {k: v for k, v in ex.items() if k != "image"}
        print(f"\nSample row — channel={chan} (image omitted):")
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
    ds = Dataset.from_list(flat, features=features)  # `flat` assembled above
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"\nSaved dataset ({len(ds)} rows) -> {out}")
    print("Columns:", ds.column_names)


if __name__ == "__main__":
    main()
