#!/usr/bin/env python3
"""
Post-hoc mismatch rescore using reasoning-trace analysis.

For trials already classified as 'neither' by exact-match scoring,
this script inspects the model's reasoning chain to infer which problem
it was actually attempting to solve.

New categories added on top of existing ones:
  text_reasoning  — neither by answer, but reasoning mentions text problem content
  image_reasoning — neither by answer, but reasoning mentions image problem content
  neither         — neither answer nor reasoning points to either problem

Usage:
    python scripts/rescore_mismatch_reasoning.py
    python scripts/rescore_mismatch_reasoning.py --results-dir path/to/results
    python scripts/rescore_mismatch_reasoning.py --model Qwen2-VL-2B-Instruct
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Keyword extraction ────────────────────────────────────────────────────────

def extract_keywords(question: str) -> set:
    """
    Extract distinctive keywords from a question.
    Filters out common stop words and short tokens.
    Returns lowercase strings.
    """
    STOP = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need",
        "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "into", "from", "up", "out", "as", "if", "or", "and", "but",
        "not", "no", "so", "it", "its", "he", "she", "they", "we",
        "you", "i", "his", "her", "their", "our", "how", "what",
        "many", "much", "more", "each", "every", "per", "total",
        "find", "much", "does", "make", "take", "get", "give",
        "how", "long", "old", "new", "all", "any", "some", "than",
    }
    tokens = re.findall(r"[a-zA-Z]+", question.lower())
    return {t for t in tokens if len(t) > 3 and t not in STOP}


def extract_numbers(text: str) -> set:
    """Extract all numbers appearing in text as strings."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def score_by_reasoning(prediction: str, image_question: str,
                       text_question: str) -> str:
    """
    Infer which problem the model was attempting based on reasoning content.

    Strategy:
    1. Count keyword hits from each question in the prediction text
    2. Count number hits from each question in the prediction text
    3. Combine scores — keyword overlap is weighted more than number overlap
       since numbers can appear coincidentally

    Returns: 'text_reasoning', 'image_reasoning', or 'neither'
    """
    pred_lower = prediction.lower()
    pred_numbers = extract_numbers(prediction)

    img_keywords = extract_keywords(image_question)
    txt_keywords = extract_keywords(text_question)

    # Unique keywords — only count words that appear in one question but not both
    # This avoids common math words ("cost", "total") inflating both scores
    img_unique = img_keywords - txt_keywords
    txt_unique = txt_keywords - img_keywords

    img_kw_hits = sum(1 for kw in img_unique if kw in pred_lower)
    txt_kw_hits = sum(1 for kw in txt_unique if kw in pred_lower)

    # Number overlap — numbers mentioned in the question appearing in reasoning
    img_nums = extract_numbers(image_question)
    txt_nums = extract_numbers(text_question)
    img_unique_nums = img_nums - txt_nums
    txt_unique_nums = txt_nums - img_nums

    img_num_hits = len(pred_numbers & img_unique_nums)
    txt_num_hits = len(pred_numbers & txt_unique_nums)

    # Combined score: keywords weighted 2x over numbers
    img_score = img_kw_hits * 2 + img_num_hits
    txt_score = txt_kw_hits * 2 + txt_num_hits

    if txt_score > img_score and txt_score > 0:
        return "text_reasoning"
    if img_score > txt_score and img_score > 0:
        return "image_reasoning"
    return "neither"


# ── Rescore ───────────────────────────────────────────────────────────────────

def rescore_model(model_dir: str) -> dict:
    """Rescore mismatch_results.csv for one model directory."""
    mm_path = os.path.join(model_dir, "mismatch_results.csv")
    if not os.path.exists(mm_path):
        print(f"  No mismatch_results.csv in {model_dir} — skipping")
        return {}

    mm = pd.read_csv(mm_path)
    model_name = os.path.basename(model_dir)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    # Original distribution
    orig_counts = mm["follows"].value_counts().to_dict()
    print(f"\nOriginal follows distribution:")
    for k, v in sorted(orig_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v}")

    # Only rescore 'neither' rows
    neither_mask = mm["follows"] == "neither"
    n_neither = neither_mask.sum()
    print(f"\nRescoring {n_neither} 'neither' rows by reasoning trace...")

    rescored = []
    for _, row in mm[neither_mask].iterrows():
        new_label = score_by_reasoning(
            str(row["prediction"]),
            str(row["image_question"]),
            str(row["text_question"]),
        )
        rescored.append(new_label)

    mm.loc[neither_mask, "follows_rescored"] = rescored

    # For non-neither rows, follows_rescored = follows
    mm.loc[~neither_mask, "follows_rescored"] = mm.loc[~neither_mask, "follows"]

    # New distribution
    new_counts = mm["follows_rescored"].value_counts().to_dict()
    print(f"\nRescored follows distribution:")
    for k, v in sorted(new_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v}")

    # Preference rates (decidable = exact text + exact image + reasoning versions)
    n_text = new_counts.get("text", 0) + new_counts.get("text_reasoning", 0)
    n_image = new_counts.get("image", 0) + new_counts.get("image_reasoning", 0)
    n_neither_final = new_counts.get("neither", 0)
    n_total = len(mm)
    decidable = n_text + n_image

    print(f"\n-- Preference (exact + reasoning combined) --")
    print(f"  Text  (exact+reasoning) : {n_text}  ({n_text/n_total*100:.1f}% of all)")
    print(f"  Image (exact+reasoning) : {n_image}  ({n_image/n_total*100:.1f}% of all)")
    print(f"  True neither            : {n_neither_final}  ({n_neither_final/n_total*100:.1f}% of all)")
    if decidable > 0:
        print(f"  Text preference (decidable): {n_text/decidable:.3f}")
        print(f"  Image preference (decidable): {n_image/decidable:.3f}")

    # Breakdown of what the 'neither' rows became
    rescore_counts = Counter(rescored)
    print(f"\n-- Of the {n_neither} original 'neither' rows --")
    for k, v in sorted(rescore_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v}  ({v/n_neither*100:.1f}%)")

    # Save rescored CSV
    out_path = os.path.join(model_dir, "mismatch_results_rescored.csv")
    mm.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Save rescored statistics
    stats = {
        "model": model_name,
        "n_total": n_total,
        "original_counts": orig_counts,
        "rescored_counts": new_counts,
        "neither_breakdown": dict(rescore_counts),
        "text_total": n_text,
        "image_total": n_image,
        "neither_final": n_neither_final,
        "decidable": decidable,
        "text_preference_rescored": n_text / decidable if decidable else None,
        "image_preference_rescored": n_image / decidable if decidable else None,
        "neither_rate_rescored": n_neither_final / n_total if n_total else None,
    }
    stats_path = os.path.join(model_dir, "mismatch_rescore_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    # Merge rescore fields into statistics.json under a 'rescore' key
    stats_json_path = os.path.join(model_dir, "statistics.json")
    if os.path.exists(stats_json_path):
        with open(stats_json_path, encoding="utf-8") as f:
            main_stats = json.load(f)
        if "rescore" not in main_stats:
            main_stats["rescore"] = {
                "text_total": n_text,
                "image_total": n_image,
                "neither_final": n_neither_final,
                "decidable": decidable,
                "text_preference_rescored": n_text / decidable if decidable else None,
                "image_preference_rescored": n_image / decidable if decidable else None,
                "neither_rate_rescored": n_neither_final / n_total if n_total else None,
                "original_neither_count": n_neither,
                "neither_breakdown": dict(rescore_counts),
            }
            with open(stats_json_path, "w", encoding="utf-8") as f:
                json.dump(main_stats, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else x)
            print(f"Updated: {stats_json_path}")
        else:
            print(f"Skipped statistics.json update (rescore key already present)")

    # Append reasoning-trace section to existing statistics_report.txt (once only)
    report_path = os.path.join(model_dir, "statistics_report.txt")
    if os.path.exists(report_path) and "REASONING-TRACE RESCORE" not in open(report_path, encoding="utf-8").read():
        rescore_section = "\n".join([
            "",
            "=" * 70,
            "  REASONING-TRACE RESCORE (post-hoc)",
            "=" * 70,
            "  Method: For 'neither' trials, reasoning chain is checked for",
            "  unique keywords and numbers from each problem. Trials whose",
            "  reasoning overlaps with one problem are reclassified as",
            "  text_reasoning or image_reasoning.",
            "",
            f"  Original 'neither' count : {n_neither}  ({n_neither/n_total*100:.1f}% of N)",
            "",
            "  Neither breakdown (after rescore):",
        ])
        for k, v in sorted(rescore_counts.items(), key=lambda x: -x[1]):
            rescore_section += f"\n    {k:20s}: {v}  ({v/n_neither*100:.1f}%)"

        rescore_section += "\n\n  Combined preference (exact + reasoning):"
        rescore_section += f"\n    Text  (exact + reasoning) : {n_text}  ({n_text/n_total*100:.1f}% of N)"
        rescore_section += f"\n    Image (exact + reasoning) : {n_image}  ({n_image/n_total*100:.1f}% of N)"
        rescore_section += f"\n    True neither              : {n_neither_final}  ({n_neither_final/n_total*100:.1f}% of N)"
        if decidable > 0:
            rescore_section += f"\n    Text preference (decidable) : {n_text/decidable:.3f}"
            rescore_section += f"\n    Image preference (decidable): {n_image/decidable:.3f}"
        rescore_section += "\n" + "=" * 70 + "\n"

        with open(report_path, "a", encoding="utf-8") as f:
            f.write(rescore_section)
        print(f"Appended rescore section to: {report_path}")

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rescore mismatch 'neither' cases using reasoning-trace analysis")
    parser.add_argument("--results-dir", default="results/phase1",
                        help="Directory containing per-model result folders")
    parser.add_argument("--model", default=None,
                        help="Run on a single model folder name only")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    if args.model:
        model_dirs = [os.path.join(results_dir, args.model)]
    else:
        model_dirs = [
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]

    all_stats = {}
    for model_dir in sorted(model_dirs):
        stats = rescore_model(model_dir)
        if stats:
            all_stats[stats["model"]] = stats

    if len(all_stats) > 1:
        print(f"\n\n{'='*60}")
        print("  CROSS-MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Model':40s}  TextPref(orig)  TextPref(rescored)  NeitherRate(rescored)")
        for model, s in all_stats.items():
            orig_txt = s["original_counts"].get("text", 0)
            orig_img = s["original_counts"].get("image", 0)
            orig_dec = orig_txt + orig_img
            orig_pref = orig_txt / orig_dec if orig_dec else 0
            print(f"  {model:40s}  {orig_pref:.3f}           "
                  f"{s['text_preference_rescored']:.3f}               "
                  f"{s['neither_rate_rescored']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
