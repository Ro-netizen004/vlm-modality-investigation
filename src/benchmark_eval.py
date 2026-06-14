"""
Multi-benchmark evaluation logic.

Two evaluation protocols based on benchmark type:

PROTOCOL A — Text-based math (GSM8K, SVAMP, AQuA-RAT, MATH):
  Condition 1: Text-only (vision disabled)
  Condition 2: Rendered image of text (vision enabled)
  Condition 3: Modality mismatch (image_i + text_{i+1})
  → Same as original study, tests whether vision encoding hurts text reasoning.

PROTOCOL B — Visual benchmarks (MathVista, ScienceQA, AI2D, ChartQA):
  Condition 1: Image + question (standard multimodal, native use case)
  Condition 2: Text description of image + question (vision disabled)
  → Tests whether VLMs actually USE visual information or just rely on text.
  Note: Condition 2 only works for problems where image content is
  described/inferable from the question. We skip it for pure visual problems.

Both protocols produce comparable statistics (McNemar, CIs, effect sizes).
"""

import os
import re
import json
import time
import pandas as pd
from collections import Counter
from typing import List
from tqdm import tqdm

from src.benchmarks import BenchmarkItem, get_benchmark_info, extract_number_from_answer
from src.evaluation import (
    answers_match, classify_error, compute_accuracy,
    compute_all_statistics, format_statistics_report,
    extract_numeric_answer, mcnemar_test, bootstrap_ci,
    binomial_ci, cohens_h,
)
from src.rendering import render_text_to_image, render_all_images, load_image
from src.visualization import plot_error_breakdown, plot_mismatch_dominance


# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER MATCHING — supports both numeric and multiple-choice
# ══════════════════════════════════════════════════════════════════════════════

def match_answer(prediction: str, item: BenchmarkItem) -> bool:
    """
    Match prediction against reference, handling multiple formats.
    - Numeric answers: round and compare
    - Multiple choice: extract letter (A/B/C/D/E)
    - String answers: exact or relaxed string match
    """
    if not prediction or not prediction.strip():
        return False

    # Multiple-choice: check if prediction contains the correct letter
    if item.choices is not None and item.correct_choice is not None:
        correct_letter = chr(65 + item.correct_choice)  # A, B, C, D...
        # Look for standalone letter in prediction
        letters_found = re.findall(r'\b([A-E])\b', prediction.upper())
        if letters_found:
            return letters_found[-1] == correct_letter
        # Also check "#### A" format
        canon = re.search(r"####\s*([A-E])", prediction.upper())
        if canon:
            return canon.group(1) == correct_letter
        return False

    # Numeric comparison
    # TODO: Phase 3 — round() is correct for GSM8K (integer answers) but wrong for
    # SVAMP (decimals) and MATH (fractions/expressions). Replace with per-dataset
    # tolerance: exact float for SVAMP, expression evaluator for MATH.
    if item.reference_number is not None:
        pred_num = extract_numeric_answer(prediction)
        if pred_num is not None:
            return round(pred_num) == round(item.reference_number)

    # Fallback: standard numeric match against reference_answer string
    return answers_match(prediction, item.reference_answer)


def classify_benchmark_error(prediction: str, item: BenchmarkItem) -> str:
    """Classify error type, handling multiple-choice and numeric."""
    if match_answer(prediction, item):
        return "correct"
    if item.choices is not None:
        # For MC: wrong_choice or no_answer
        letters_found = re.findall(r'\b([A-E])\b', prediction.upper())
        return "wrong_choice" if letters_found else "no_answer"
    return classify_error(prediction, item.reference_answer)


# ══════════════════════════════════════════════════════════════════════════════
#  PROTOCOL A — Text-based benchmarks (render text as images)
# ══════════════════════════════════════════════════════════════════════════════

def run_protocol_a(model, items: List[BenchmarkItem], benchmark_name: str,
                   output_dir: str, image_dir: str = None):
    """
    Protocol A for text-based math benchmarks.
    3 conditions: text-only, rendered image, mismatch.
    """
    n = len(items)
    questions = [item.question for item in items]

    # Render images if needed
    if image_dir is None:
        image_dir = os.path.join(output_dir, f"{benchmark_name}_images")
    render_all_images(questions, image_dir)

    results = {
        "text_preds": [], "text_correct": [], "text_errors": [],
        "img_preds": [], "img_correct": [], "img_errors": [],
        "mm_preds": [], "mm_follows": [], "mm_img_diffs": [], "mm_txt_diffs": [],
    }

    # ── Condition 1: Text-Only ──
    print(f"\n[{benchmark_name}] Condition 1: Text-Only")
    for item in tqdm(items, desc="Text"):
        try:
            pred = model.generate_text_only(item.question)
        except Exception as e:
            pred = f"ERROR: {e}"
        results["text_preds"].append(pred)
        results["text_correct"].append(match_answer(pred, item))
        results["text_errors"].append(classify_benchmark_error(pred, item))
    print(f"  Accuracy: {compute_accuracy(results['text_correct']):.3f}")

    # ── Condition 2: Rendered Image ──
    print(f"\n[{benchmark_name}] Condition 2: Rendered Image")
    for i, item in enumerate(tqdm(items, desc="Image")):
        try:
            img = load_image(i, image_dir)
            pred = model.generate_with_image(img)
        except Exception as e:
            pred = f"ERROR: {e}"
        results["img_preds"].append(pred)
        results["img_correct"].append(match_answer(pred, item))
        results["img_errors"].append(classify_benchmark_error(pred, item))
    print(f"  Accuracy: {compute_accuracy(results['img_correct']):.3f}")

    # ── Condition 3: Mismatch ──
    print(f"\n[{benchmark_name}] Condition 3: Mismatch")
    for i, item in enumerate(tqdm(items, desc="Mismatch")):
        txt_idx = (i + 1) % n
        txt_item = items[txt_idx]
        prompt = (
            f"Solve the following math problem step by step. "
            f"End with '#### <answer>'.\n\nProblem: {txt_item.question}"
        )
        try:
            img = load_image(i, image_dir)
            pred = model.generate_with_image(img, text_prompt=prompt)
        except Exception as e:
            pred = f"ERROR: {e}"

        pred_val = extract_numeric_answer(pred)
        img_val = item.reference_number
        txt_val = txt_item.reference_number

        if pred_val is None or img_val is None or txt_val is None:
            follows, d_img, d_txt = "invalid", None, None
        else:
            d_img = abs(pred_val - img_val)
            d_txt = abs(pred_val - txt_val)
            follows = "image" if d_img < d_txt else ("text" if d_txt < d_img else "equal")

        results["mm_preds"].append(pred)
        results["mm_follows"].append(follows)
        results["mm_img_diffs"].append(d_img)
        results["mm_txt_diffs"].append(d_txt)

    print(f"  Dominance: {Counter(results['mm_follows'])}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PROTOCOL B — Visual benchmarks (native images)
# ══════════════════════════════════════════════════════════════════════════════

def run_protocol_b(model, items: List[BenchmarkItem], benchmark_name: str,
                   output_dir: str):
    """
    Protocol B for visual benchmarks that come with native images.
    Condition 1: Image + question (multimodal — standard use case)
    Condition 2: Text-only question (vision disabled — baseline)
    Compares whether the model actually uses visual information.
    """
    # Filter to items that have images
    visual_items = [item for item in items if item.image is not None]
    text_items = [item for item in items if item.image is None]

    print(f"\n[{benchmark_name}] {len(visual_items)} items with images, "
          f"{len(text_items)} text-only")

    results = {
        "multimodal_preds": [], "multimodal_correct": [], "multimodal_errors": [],
        "text_preds": [], "text_correct": [], "text_errors": [],
    }

    all_items = visual_items  # evaluate on visual items

    # ── Condition 1: Image + Question (multimodal) ──
    print(f"\n[{benchmark_name}] Condition 1: Image + Question")
    for item in tqdm(all_items, desc="Multimodal"):
        prompt = item.question
        if item.choices:
            prompt += "\nAnswer with the letter of the correct option."
        else:
            prompt += "\nProvide the answer. End with '#### <answer>'."
        try:
            pred = model.generate_with_image(item.image, text_prompt=prompt)
        except Exception as e:
            pred = f"ERROR: {e}"
        results["multimodal_preds"].append(pred)
        results["multimodal_correct"].append(match_answer(pred, item))
        results["multimodal_errors"].append(classify_benchmark_error(pred, item))
    print(f"  Accuracy: {compute_accuracy(results['multimodal_correct']):.3f}")

    # ── Condition 2: Text-only (no image) ──
    print(f"\n[{benchmark_name}] Condition 2: Text-Only (no image)")
    for item in tqdm(all_items, desc="Text-Only"):
        prompt = item.question
        if item.choices:
            prompt += "\nAnswer with the letter of the correct option."
        else:
            prompt += "\nProvide the answer. End with '#### <answer>'."
        try:
            pred = model.generate_text_only(prompt)
        except Exception as e:
            pred = f"ERROR: {e}"
        results["text_preds"].append(pred)
        results["text_correct"].append(match_answer(pred, item))
        results["text_errors"].append(classify_benchmark_error(pred, item))
    print(f"  Accuracy: {compute_accuracy(results['text_correct']):.3f}")

    return results, all_items


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_protocol_a_results(results, items, model_name, benchmark_name, output_dir):
    """Save Protocol A results with statistics."""
    short = model_name.split("/")[-1]
    save_dir = os.path.join(output_dir, short, benchmark_name)
    os.makedirs(save_dir, exist_ok=True)

    n = len(items)

    # Stats
    stats = compute_all_statistics(
        results["text_correct"], results["img_correct"], results["mm_follows"])

    report = format_statistics_report(stats)
    print(report)

    # CSV
    df = pd.DataFrame({
        "problem_id": [item.id for item in items],
        "question": [item.question for item in items],
        "reference": [item.reference_answer for item in items],
        "pred_text": results["text_preds"],
        "correct_text": results["text_correct"],
        "error_text": results["text_errors"],
        "pred_rendered": results["img_preds"],
        "correct_rendered": results["img_correct"],
        "error_rendered": results["img_errors"],
        "pred_mismatch": results["mm_preds"],
        "mismatch_follows": results["mm_follows"],
    })
    df.to_csv(os.path.join(save_dir, "results.csv"), index=False)

    # Statistics
    with open(os.path.join(save_dir, "statistics.json"), "w") as f:
        serializable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in stats.items()}
        serializable["benchmark"] = benchmark_name
        serializable["model"] = model_name
        json.dump(serializable, f, indent=2)

    with open(os.path.join(save_dir, "statistics_report.txt"), "w") as f:
        f.write(report)

    # Plots
    plot_error_breakdown(
        {"Text-Only": results["text_errors"], "Rendered Image": results["img_errors"]},
        f"{short} — {benchmark_name}", save_dir)
    plot_mismatch_dominance(
        Counter(results["mm_follows"]), f"{short} — {benchmark_name}", save_dir)

    print(f"Saved: {save_dir}")
    return stats


def save_protocol_b_results(results, items, model_name, benchmark_name, output_dir):
    """Save Protocol B results with statistics."""
    short = model_name.split("/")[-1]
    save_dir = os.path.join(output_dir, short, benchmark_name)
    os.makedirs(save_dir, exist_ok=True)

    # Stats (use multimodal as "text_correct" and text-only as "img_correct"
    # for McNemar comparison — labels are just for the paired test)
    mc = results["multimodal_correct"]
    tc = results["text_correct"]
    n = len(mc)

    acc_mm = compute_accuracy(mc)
    acc_text = compute_accuracy(tc)
    mm_ci = binomial_ci(sum(mc), n)
    text_ci = binomial_ci(sum(tc), n)
    mm_boot = bootstrap_ci(mc)
    text_boot = bootstrap_ci(tc)
    mcn_chi2, mcn_p, mcn_b, mcn_c = mcnemar_test(mc, tc)
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

    # Report
    lines = [
        "=" * 70,
        f"  {benchmark_name.upper()} — PROTOCOL B RESULTS",
        "=" * 70,
        f"  Model: {model_name}",
        f"  N = {n}",
        "",
        f"  Multimodal (img+text) : {acc_mm:.3f}  95% CI [{mm_ci[0]:.3f}, {mm_ci[1]:.3f}]",
        f"  Text-Only (no image) : {acc_text:.3f}  95% CI [{text_ci[0]:.3f}, {text_ci[1]:.3f}]",
        f"  Difference           : {acc_mm - acc_text:+.3f}",
        "",
        f"  McNemar p-value : {mcn_p:.6f}  {'***' if mcn_p<0.001 else '**' if mcn_p<0.01 else '*' if mcn_p<0.05 else 'ns'}",
        f"  Cohen's h       : {effect:.3f}",
        "",
        "  Interpretation:",
    ]
    if acc_mm > acc_text and mcn_p < 0.05:
        lines.append("    Vision HELPS — model uses visual info for better answers.")
    elif acc_text > acc_mm and mcn_p < 0.05:
        lines.append("    Vision HURTS — model performs worse with images.")
    else:
        lines.append("    No significant difference between modalities.")
    lines.append("=" * 70)
    report = "\n".join(lines)
    print(report)

    # Save
    df = pd.DataFrame({
        "problem_id": [item.id for item in items],
        "question": [item.question for item in items],
        "reference": [item.reference_answer for item in items],
        "pred_multimodal": results["multimodal_preds"],
        "correct_multimodal": results["multimodal_correct"],
        "pred_text": results["text_preds"],
        "correct_text": results["text_correct"],
    })
    df.to_csv(os.path.join(save_dir, "results.csv"), index=False)

    with open(os.path.join(save_dir, "statistics.json"), "w") as f:
        serializable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in stats.items()}
        json.dump(serializable, f, indent=2)

    with open(os.path.join(save_dir, "statistics_report.txt"), "w") as f:
        f.write(report)

    plot_error_breakdown(
        {"Multimodal": results["multimodal_errors"], "Text-Only": results["text_errors"]},
        f"{short} — {benchmark_name}", save_dir)

    print(f"Saved: {save_dir}")
    return stats
