"""Answer extraction, matching, error classification, and statistical testing."""

import re
import numpy as np
from collections import Counter
from scipy import stats


# ── Answer Extraction ────────────────────────────────────────────────────────

def extract_numeric_answer(text: str):
    """Extract the final numeric answer from model output or GSM8K reference."""
    if not text or not text.strip():
        return None
    canon = re.search(r"####\s*([\-\d,\.]+)", text)
    if canon:
        try:
            return float(canon.group(1).replace(",", ""))
        except ValueError:
            pass
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    numbers = [n for n in numbers if n.replace(",", "").replace(".", "").replace("-", "")]
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def answers_match(pred: str, ref: str) -> bool:
    """Compare predicted and reference answers numerically (rounded to int)."""
    # TODO: Phase 3 — add per-dataset tolerance (e.g. exact float for SVAMP decimals,
    # letter matching for AQuA-RAT, expression matching for MATH).
    # See vlm_benchmark/answer_parsing.py for the strategy pattern to port.
    p = extract_numeric_answer(pred)
    r = extract_numeric_answer(ref)
    if p is None or r is None:
        return False
    return round(p) == round(r)


def numeric_distance(pred: str, ref: str) -> float:
    """Absolute numeric distance between prediction and reference."""
    p = extract_numeric_answer(pred)
    r = extract_numeric_answer(ref)
    if p is None or r is None:
        return float("inf")
    return abs(p - r)


def score_mismatch_follows(prediction: str, image_ref: str, text_ref: str) -> str:
    """
    Correctness-based mismatch classification (GSM8K numeric answers).

    Returns one of: image, text, neither, ambiguous, invalid.
    """
    pred_val = extract_numeric_answer(str(prediction))
    img_val = extract_numeric_answer(str(image_ref))
    txt_val = extract_numeric_answer(str(text_ref))

    if pred_val is None:
        return "invalid"
    if img_val is None or txt_val is None:
        return "invalid"
    if img_val == txt_val:
        return "ambiguous"
    # round() is fine for GSM8K integers; extend for decimals/MC in multi-benchmark
    if round(pred_val) == round(img_val):
        return "image"
    if round(pred_val) == round(txt_val):
        return "text"
    return "neither"


# ── Error Classification ─────────────────────────────────────────────────────

VISION_KEYWORDS = [
    "cannot read", "can't read", "image is unclear",
    "unable to see", "blurry", "illegible", "not visible",
]


def classify_error(prediction: str, reference: str) -> str:
    """Classify error type: correct | no_number | vision_error | arithmetic_error | reasoning_error."""
    if answers_match(prediction, reference):
        return "correct"
    pred_num = extract_numeric_answer(prediction)
    if pred_num is None:
        return "no_number"
    if any(kw in prediction.lower() for kw in VISION_KEYWORDS):
        return "vision_error"
    has_steps = bool(re.search(r"[\+\-\*\/\=]", prediction))
    return "arithmetic_error" if has_steps else "reasoning_error"


# ── Accuracy ──────────────────────────────────────────────────────────────────

def compute_accuracy(correct_flags):
    return sum(correct_flags) / len(correct_flags) if correct_flags else 0.0


def error_counts(errors, categories=None):
    if categories is None:
        categories = ["correct", "arithmetic_error", "reasoning_error", "no_number", "vision_error"]
    c = Counter(errors)
    return {cat: c.get(cat, 0) for cat in categories}


# ── Statistical Testing ──────────────────────────────────────────────────────

def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test for paired binary outcomes.
    Uses exact binomial test for small discordant counts and Yates-corrected
    chi-square for larger counts.
    Returns: chi2 statistic, p-value, b, c (discordant counts).
    """
    assert len(correct_a) == len(correct_b)

    a = [bool(x) for x in correct_a]
    b_flags = [bool(x) for x in correct_b]

    n_cc = sum(ai and bi     for ai, bi in zip(a, b_flags))  # both correct
    n_cw = sum(ai and not bi for ai, bi in zip(a, b_flags))  # A correct, B wrong (c)
    n_wc = sum(not ai and bi for ai, bi in zip(a, b_flags))  # A wrong, B correct (b)
    n_ww = sum(not ai and not bi for ai, bi in zip(a, b_flags))  # both wrong

    b = n_wc  # discordant: only B correct
    c = n_cw  # discordant: only A correct

    if b + c == 0:
        return 0.0, 1.0, b, c

    chi2_val = (abs(b - c) - 1) ** 2 / (b + c)
    if b + c < 25:
        p_value = stats.binomtest(min(b, c), n=b + c, p=0.5).pvalue
    else:
        p_value = 1 - stats.chi2.cdf(chi2_val, df=1)

    return float(chi2_val), float(p_value), b, c


def bootstrap_ci(correct_flags, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    arr = np.array(correct_flags, dtype=float)
    n = len(arr)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    lower = np.percentile(means, 100 * alpha)
    upper = np.percentile(means, 100 * (1 - alpha))
    return lower, upper


def binomial_ci(k, n, ci=0.95):
    """Exact (Clopper-Pearson) confidence interval for a proportion."""
    alpha = 1 - ci
    lower = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lower, upper


def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def chi_squared_test(observed_counts):
    """
    Chi-squared test for modality preference on DECIDABLE mismatch trials only.
    Only 'image' and 'text' counts are tested — 'neither', 'ambiguous', 'equal',
    and 'invalid' are excluded because they don't indicate a clear modality preference.
    Tests against uniform distribution (no preference between image and text).
    """
    image = observed_counts.get("image", 0)
    text  = observed_counts.get("text", 0)
    decidable = image + text

    if decidable == 0:
        return 0.0, 1.0

    total = sum(observed_counts.values())
    excluded = total - decidable
    if excluded > 0:
        print(f"  Note: chi-squared run on {decidable}/{total} decidable trials "
              f"({excluded} excluded: neither/ambiguous/equal/invalid)")

    chi2, p_value = stats.chisquare([image, text])
    return float(chi2), float(p_value)


def compute_all_statistics(text_correct, img_correct, mismatch_follows):
    """Compute all statistical tests and return a summary dict."""
    n = len(text_correct)
    acc_text = compute_accuracy(text_correct)
    acc_img = compute_accuracy(img_correct)

    text_ci = binomial_ci(sum(text_correct), n)
    img_ci = binomial_ci(sum(img_correct), n)

    text_boot_ci = bootstrap_ci(text_correct)
    img_boot_ci = bootstrap_ci(img_correct)

    mcn_chi2, mcn_p, mcn_b, mcn_c = mcnemar_test(text_correct, img_correct)
    effect = cohens_h(acc_text, acc_img)

    follow_counts = Counter(mismatch_follows)
    chi2_mm, p_mm = chi_squared_test(follow_counts)

    return {
        "n": n,
        "acc_text": acc_text,
        "acc_img": acc_img,
        "acc_drop": acc_img - acc_text,
        "text_ci_95": text_ci,
        "img_ci_95": img_ci,
        "text_boot_ci_95": text_boot_ci,
        "img_boot_ci_95": img_boot_ci,
        "mcnemar_chi2": mcn_chi2,
        "mcnemar_p": mcn_p,
        "mcnemar_b": mcn_b,
        "mcnemar_c": mcn_c,
        "cohens_h": effect,
        "mismatch_chi2": chi2_mm,
        "mismatch_p": p_mm,
        "mismatch_counts": dict(follow_counts),
    }


def format_statistics_report(stats_dict: dict) -> str:
    """Format statistics into a human-readable report."""
    s = stats_dict
    lines = [
        "=" * 70,
        "  STATISTICAL ANALYSIS REPORT",
        "=" * 70,
        f"  Sample size: N = {s['n']}",
        "",
        "  ── Accuracy ──────────────────────────────────────────",
        f"  Text-Only      : {s['acc_text']:.3f}  95% CI [{s['text_ci_95'][0]:.3f}, {s['text_ci_95'][1]:.3f}]",
        f"  Rendered Image : {s['acc_img']:.3f}  95% CI [{s['img_ci_95'][0]:.3f}, {s['img_ci_95'][1]:.3f}]",
        f"  Drop           : {s['acc_drop']:+.3f}",
        "",
        "  ── Bootstrap 95% CI ──────────────────────────────────",
        f"  Text-Only      : [{s['text_boot_ci_95'][0]:.3f}, {s['text_boot_ci_95'][1]:.3f}]",
        f"  Rendered Image : [{s['img_boot_ci_95'][0]:.3f}, {s['img_boot_ci_95'][1]:.3f}]",
        "",
        "  ── McNemar's Test (paired comparison) ───────────────",
        f"  Text correct & Image wrong (b): {s['mcnemar_b']}",
        f"  Text wrong & Image correct (c): {s['mcnemar_c']}",
        f"  Chi-squared (Yates)           : {s['mcnemar_chi2']:.3f}",
        f"  p-value                       : {s['mcnemar_p']:.6f}",
        f"  Significant (p < 0.05)        : {'YES' if s['mcnemar_p'] < 0.05 else 'NO'}",
        "",
        f"  ── Effect Size ───────────────────────────────────────",
        f"  Cohen's h : {s['cohens_h']:.3f}  ({'small' if abs(s['cohens_h']) < 0.5 else 'medium' if abs(s['cohens_h']) < 0.8 else 'large'})",
        "",
        "  ── Mismatch Modality Preference ─────────────────────",
        f"  Follows image : {s['mismatch_counts'].get('image', 0)}",
        f"  Follows text  : {s['mismatch_counts'].get('text', 0)}",
        f"  Neither       : {s['mismatch_counts'].get('neither', 0)}",
        f"  Ambiguous     : {s['mismatch_counts'].get('ambiguous', 0)}",
        f"  Invalid       : {s['mismatch_counts'].get('invalid', 0)}",
        f"  Chi-squared   : {s['mismatch_chi2']:.3f}",
        f"  p-value       : {s['mismatch_p']:.6f}",
        f"  Significant   : {'YES' if s['mismatch_p'] < 0.05 else 'NO'}",
        "=" * 70,
    ]
    return "\n".join(lines)
