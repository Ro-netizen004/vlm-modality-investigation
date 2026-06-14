#!/usr/bin/env python3
"""
Rescore mismatch condition for already-completed model runs.

Fixes the distance-based mismatch classification bug by replacing it with
correctness-based classification: image / text / neither / ambiguous / invalid.

Usage:
    python scripts/rescore_mismatch.py --csv path/to/gsm8k_results.csv
    python scripts/rescore_mismatch.py --csv path/to/gsm8k_results.csv --dry-run

Overwrites mismatch_follows in the CSV and regenerates statistics.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    extract_numeric_answer,
    compute_all_statistics,
    format_statistics_report,
)
from src.visualization import plot_error_breakdown, plot_mismatch_dominance


def rescore_row(pred: str, img_ref: str, txt_ref: str) -> str:
    """
    Correctness-based mismatch classification.

    image     — prediction matches image problem's answer
    text      — prediction matches text problem's answer
    neither   — prediction matches neither (model was wrong for both)
    ambiguous — image and text answers are the same (can't distinguish)
    invalid   — prediction or a reference answer has no extractable number
    """
    pred_val = extract_numeric_answer(str(pred))
    img_val  = extract_numeric_answer(str(img_ref))
    txt_val  = extract_numeric_answer(str(txt_ref))

    if pred_val is None:
        return "invalid"
    if img_val is None or txt_val is None:
        return "invalid"
    if img_val == txt_val:
        return "ambiguous"
    if round(pred_val) == round(img_val):
        return "image"
    if round(pred_val) == round(txt_val):
        return "text"
    return "neither"


def rescore_csv(csv_path: Path, dry_run: bool = False) -> None:
    df = pd.read_csv(csv_path)
    n = len(df)

    required = {"problem_id", "reference", "pred_mismatch",
                "correct_text", "correct_rendered", "mismatch_follows"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing columns: {missing_cols}")

    print(f"\nRescoring {n} rows from: {csv_path}")
    print(f"\nOLD mismatch distribution:")
    print(dict(Counter(df["mismatch_follows"])))

    # Build new mismatch_follows
    new_follows = []
    for i, row in df.iterrows():
        pid      = int(row["problem_id"])
        txt_idx  = (pid + 1) % n
        img_ref  = row["reference"]                        # reference for image problem
        txt_ref  = df.loc[txt_idx, "reference"]           # reference for text problem (next row)
        pred     = row["pred_mismatch"]
        new_follows.append(rescore_row(pred, img_ref, txt_ref))

    print(f"\nNEW mismatch distribution:")
    print(dict(Counter(new_follows)))

    # Show what changed
    changed = sum(a != b for a, b in zip(df["mismatch_follows"], new_follows))
    print(f"\nRows where label changed: {changed}/{n} ({changed/n*100:.1f}%)")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Overwrite mismatch_follows, drop old diff columns if present
    df["mismatch_follows"] = new_follows
    df.drop(columns=["mismatch_img_diff", "mismatch_txt_diff"], errors="ignore", inplace=True)

    df.to_csv(csv_path, index=False)
    print(f"\nSaved corrected CSV: {csv_path}")

    # Recompute statistics
    text_correct     = df["correct_text"].tolist()
    img_correct      = df["correct_rendered"].tolist()
    mismatch_follows = df["mismatch_follows"].tolist()

    stats = compute_all_statistics(text_correct, img_correct, mismatch_follows)
    print(format_statistics_report(stats))

    # Save statistics.json alongside the CSV
    stats_path = csv_path.parent / "statistics.json"
    serializable = {k: list(v) if isinstance(v, tuple) else v
                    for k, v in stats.items()}
    with open(stats_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved statistics: {stats_path}")

    # Save statistics_report.txt
    report_path = csv_path.parent / "statistics_report.txt"
    with open(report_path, "w") as f:
        f.write(format_statistics_report(stats))
    print(f"Saved report: {report_path}")

    # Regenerate plots
    model_name = csv_path.parent.name
    output_dir = str(csv_path.parent)

    plot_error_breakdown(
        {"Text-Only": df["error_text"].tolist(),
         "Rendered Image": df["error_rendered"].tolist()},
        model_name,
        output_dir,
    )

    plot_mismatch_dominance(
        Counter(mismatch_follows),
        model_name,
        output_dir,
    )

    print(f"Regenerated plots in: {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Rescore mismatch condition in gsm8k_results.csv")
    p.add_argument("--csv", required=True, help="Path to gsm8k_results.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change without writing files")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rescore_csv(csv_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
