"""
Deep error analysis utilities.

Goes beyond simple error classification to understand:
1. Problem difficulty factors (number of reasoning steps, numerical magnitude)
2. Disagreement analysis (text-correct/image-wrong and vice versa)
3. Error propagation (where in the chain of reasoning errors occur)
4. Correlation analysis (what predicts modality gap?)
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional

from src.evaluation import extract_numeric_answer, answers_match


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM DIFFICULTY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def count_reasoning_steps(reference_answer) -> int:
    """Count reasoning steps in a GSM8K reference answer (lines with calculations)."""
    reference_answer = str(reference_answer) if reference_answer is not None else ""
    if not reference_answer:
        return 0
    lines = reference_answer.strip().split("\n")
    step_lines = [l for l in lines if re.search(r"[\+\-\*\/\=]", l) and "####" not in l]
    return len(step_lines)


def count_numbers_in_question(question: str) -> int:
    """Count distinct numbers mentioned in the problem."""
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", question)
    return len(numbers)


def get_answer_magnitude(reference_answer: str) -> Optional[float]:
    """Get the magnitude (absolute value) of the reference answer."""
    num = extract_numeric_answer(reference_answer)
    if num is not None:
        return abs(num)
    return None


def get_question_length(question: str) -> int:
    """Word count of the question."""
    return len(question.split())


def has_multi_step_operations(reference_answer) -> bool:
    """Check if answer requires multiple different operations."""
    reference_answer = str(reference_answer) if reference_answer is not None else ""
    ops = set()
    if "+" in reference_answer or "add" in reference_answer.lower():
        ops.add("add")
    if "-" in reference_answer or "subtract" in reference_answer.lower():
        ops.add("sub")
    if "*" in reference_answer or "×" in reference_answer:
        ops.add("mul")
    if "/" in reference_answer or "÷" in reference_answer:
        ops.add("div")
    return len(ops) >= 2


def extract_problem_features(question: str, reference: str) -> Dict:
    """Extract all difficulty features for a problem."""
    return {
        "num_steps": count_reasoning_steps(reference),
        "num_numbers": count_numbers_in_question(question),
        "answer_magnitude": get_answer_magnitude(reference),
        "question_length": get_question_length(question),
        "multi_operation": has_multi_step_operations(reference),
        "has_fractions": bool(re.search(r"\d+/\d+", question)),
        "has_percentages": "%" in question or "percent" in question.lower(),
        "has_comparison": any(w in question.lower() for w in
                             ["more than", "less than", "twice", "half", "difference"]),
    }


def compute_problem_features_df(questions: List[str], references: List[str]) -> pd.DataFrame:
    """Compute features for all problems as a DataFrame."""
    features = [extract_problem_features(q, r) for q, r in zip(questions, references)]
    return pd.DataFrame(features)


# ══════════════════════════════════════════════════════════════════════════════
#  DISAGREEMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_disagreements(text_correct: List[bool], img_correct: List[bool],
                          questions: List[str], references: List[str],
                          text_preds: List[str] = None,
                          img_preds: List[str] = None) -> Dict:
    """
    Analyze problems where text-only and image conditions disagree.

    Returns:
        - counts of each disagreement type
        - indices of each type
        - feature analysis of each group
    """
    n = len(text_correct)
    assert len(img_correct) == n

    # Categorize
    both_correct = [i for i in range(n) if text_correct[i] and img_correct[i]]
    both_wrong = [i for i in range(n) if not text_correct[i] and not img_correct[i]]
    text_only_correct = [i for i in range(n) if text_correct[i] and not img_correct[i]]
    img_only_correct = [i for i in range(n) if not text_correct[i] and img_correct[i]]

    # Feature analysis for each group
    features = compute_problem_features_df(questions, references)

    def group_stats(indices):
        if not indices:
            return {}
        sub = features.iloc[indices]
        return {
            "count": len(indices),
            "avg_steps": sub["num_steps"].mean(),
            "avg_numbers": sub["num_numbers"].mean(),
            "avg_question_length": sub["question_length"].mean(),
            "avg_answer_magnitude": sub["answer_magnitude"].dropna().mean(),
            "pct_multi_operation": sub["multi_operation"].mean() * 100,
            "pct_has_percentages": sub["has_percentages"].mean() * 100,
        }

    result = {
        "both_correct": {"indices": both_correct, "stats": group_stats(both_correct)},
        "both_wrong": {"indices": both_wrong, "stats": group_stats(both_wrong)},
        "text_only_correct": {"indices": text_only_correct, "stats": group_stats(text_only_correct)},
        "img_only_correct": {"indices": img_only_correct, "stats": group_stats(img_only_correct)},
        "summary": {
            "total": n,
            "agreement_rate": (len(both_correct) + len(both_wrong)) / n,
            "text_advantage": len(text_only_correct),
            "image_advantage": len(img_only_correct),
        },
    }

    return result


def format_disagreement_report(analysis: Dict) -> str:
    """Format disagreement analysis as readable text."""
    s = analysis["summary"]
    lines = [
        "=" * 70,
        "  DISAGREEMENT ANALYSIS",
        "=" * 70,
        f"  Total problems: {s['total']}",
        f"  Agreement rate: {s['agreement_rate']:.1%}",
        f"  Text advantage (text right, image wrong): {s['text_advantage']}",
        f"  Image advantage (image right, text wrong): {s['image_advantage']}",
        "",
    ]

    for group_name, label in [
        ("both_correct", "Both Correct"),
        ("both_wrong", "Both Wrong"),
        ("text_only_correct", "Text-Only Correct (vision hurt)"),
        ("img_only_correct", "Image-Only Correct (vision helped)"),
    ]:
        g = analysis[group_name]
        stats = g["stats"]
        if not stats:
            lines.append(f"  {label}: 0 problems")
            continue
        lines.extend([
            f"  ── {label}: {stats['count']} problems ──",
            f"    Avg reasoning steps  : {stats['avg_steps']:.1f}",
            f"    Avg numbers in Q     : {stats['avg_numbers']:.1f}",
            f"    Avg question length  : {stats['avg_question_length']:.0f} words",
            f"    Avg answer magnitude : {stats.get('avg_answer_magnitude', 0):.0f}",
            f"    Multi-operation      : {stats['pct_multi_operation']:.0f}%",
            f"    Has percentages      : {stats['pct_has_percentages']:.0f}%",
            "",
        ])

    lines.append("=" * 70)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR PROPAGATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_error_propagation(prediction: str, reference: str) -> Dict:
    """
    Analyze WHERE in the reasoning chain the error occurs.

    Compares intermediate calculations in prediction vs reference step by step.
    """
    # Extract all numbers from both
    pred_numbers = [float(n.replace(",", "")) for n in re.findall(r"-?[\d,]+\.?\d*", prediction)
                    if n.replace(",", "").replace(".", "").replace("-", "")]
    ref_numbers = [float(n.replace(",", "")) for n in re.findall(r"-?[\d,]+\.?\d*", reference)
                   if n.replace(",", "").replace(".", "").replace("-", "")]

    if not pred_numbers or not ref_numbers:
        return {"error_step": -1, "total_steps": 0, "type": "no_numbers"}

    # Find first divergence point
    min_len = min(len(pred_numbers), len(ref_numbers))
    first_error = -1
    for i in range(min_len):
        if round(pred_numbers[i]) != round(ref_numbers[i]):
            first_error = i
            break

    if first_error == -1:
        if len(pred_numbers) != len(ref_numbers):
            return {
                "error_step": min_len,
                "total_steps": len(ref_numbers),
                "type": "length_mismatch",
                "pred_count": len(pred_numbers),
                "ref_count": len(ref_numbers),
            }
        return {"error_step": -1, "total_steps": len(ref_numbers), "type": "correct"}

    return {
        "error_step": first_error,
        "total_steps": len(ref_numbers),
        "type": "early_error" if first_error < len(ref_numbers) / 2 else "late_error",
        "error_position_pct": first_error / len(ref_numbers) * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_modality_gap_correlates(
    text_correct: List[bool], img_correct: List[bool],
    questions: List[str], references: List[str]
) -> pd.DataFrame:
    """
    Compute correlations between problem features and modality gap.

    For each problem, modality_gap = text_correct - img_correct
    (1 = text helped, -1 = image helped, 0 = same)
    """
    features = compute_problem_features_df(questions, references)
    features["text_correct"] = text_correct
    features["img_correct"] = img_correct
    features["modality_gap"] = [int(t) - int(i) for t, i in zip(text_correct, img_correct)]

    # Compute correlations
    numeric_cols = ["num_steps", "num_numbers", "question_length", "answer_magnitude"]
    correlations = []
    for col in numeric_cols:
        valid = features[[col, "modality_gap"]].dropna()
        if len(valid) > 10:
            from scipy.stats import pearsonr, spearmanr
            r_pearson, p_pearson = pearsonr(valid[col], valid["modality_gap"])
            r_spearman, p_spearman = spearmanr(valid[col], valid["modality_gap"])
            correlations.append({
                "feature": col,
                "pearson_r": r_pearson,
                "pearson_p": p_pearson,
                "spearman_r": r_spearman,
                "spearman_p": p_spearman,
            })

    # Binary features: point-biserial correlation
    bool_cols = ["multi_operation", "has_fractions", "has_percentages", "has_comparison"]
    for col in bool_cols:
        from scipy.stats import pointbiserialr
        valid = features[[col, "modality_gap"]].dropna()
        if len(valid) > 10:
            r, p = pointbiserialr(valid[col].astype(int), valid["modality_gap"])
            correlations.append({
                "feature": col,
                "pearson_r": r,
                "pearson_p": p,
                "spearman_r": r,  # same for binary
                "spearman_p": p,
            })

    return pd.DataFrame(correlations)
