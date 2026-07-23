"""Sensitivity of the paired arm asymmetry to answer-length normalization.

For candidate c with total continuation log-likelihood S_c and token count n_c,
score S_c / n_c**alpha for alpha in {0, .5, 1}. The existing artifacts store
S_c and its per-token mean, so n_c is recovered exactly as S_c / mean_c.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


LEVEL_FILES = {
    "image": ("level_0_clean.cll.jsonl", "level_5_heavy_degradation.cll.jsonl"),
    "text": ("level_0_clean.cll.jsonl", "level_5_heavy_corruption.cll.jsonl"),
}


def candidate_length(total, mean):
    if total is None or mean is None or not np.isfinite(total) or not np.isfinite(mean):
        return None
    if mean == 0:
        return 1 if total == 0 else None
    length = int(round(total / mean))
    if length < 1 or not np.isclose(total, mean * length, rtol=1e-5, atol=1e-5):
        return None
    return length


def load_margins(path: Path, alpha: float):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                margin = row.get("margin") or {}
                text_sum = margin.get("cll_text_sum")
                image_sum = margin.get("cll_image_sum")
                text_n = candidate_length(text_sum, margin.get("cll_text_mean"))
                image_n = candidate_length(image_sum, margin.get("cll_image_mean"))
                if text_n is None or image_n is None:
                    continue
                value = text_sum / text_n**alpha - image_sum / image_n**alpha
                if np.isfinite(value):
                    rows[int(row["i"])] = float(value)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return rows


def arm_changes(model_dir: Path, arm: str, alpha: float):
    l0_name, l5_name = LEVEL_FILES[arm]
    l0 = load_margins(model_dir / l0_name, alpha)
    l5 = load_margins(model_dir / l5_name, alpha)
    return {item: l5[item] - l0[item] for item in sorted(l0.keys() & l5.keys())}


def contrast(image_dir: Path, text_dir: Path, alpha: float):
    image = arm_changes(image_dir, "image", alpha)
    text = arm_changes(text_dir, "text", alpha)
    ids = sorted(image.keys() & text.keys())
    r_image = np.asarray([image[item] for item in ids], dtype=float)
    r_text = np.asarray([-text[item] for item in ids], dtype=float)
    return ids, r_image, r_text, r_text - r_image


def paired_stats(values, seed, resamples):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    test = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    batch = 1000
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        boot[start:start + size] = np.median(values[indices], axis=1)
    lo, hi = np.quantile(boot, (0.025, 0.975))
    return float(np.median(values)), float(lo), float(hi), float(test.pvalue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--original-image-root", type=Path,
                        default=Path("results/phase6_legibility/gsm8k"))
    parser.add_argument("--original-text-root", type=Path,
                        default=Path("results/phase7_text_legibility/gsm8k"))
    parser.add_argument("--neutral-root", type=Path,
                        default=Path("results/phase_control/role_counterbalance"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--alphas", nargs="+", type=float, default=(0.0, 0.5, 1.0))
    args = parser.parse_args()

    framings = {
        "original": (args.original_image_root, args.original_text_root),
        "neutral": (args.neutral_root / "role_neutral",
                    args.neutral_root / "text_legibility" / "role_neutral"),
    }
    print("framing\tmodel\talpha\tn\tR_image\tR_text\tasymmetry\t95% CI\tWilcoxon p")
    for framing, (image_root, text_root) in framings.items():
        for model_index, model in enumerate(args.models):
            for alpha in args.alphas:
                ids, r_image, r_text, asymmetry = contrast(
                    image_root / model, text_root / model, alpha
                )
                result = paired_stats(
                    asymmetry, seed=20260721 + model_index, resamples=args.resamples
                )
                if result is None:
                    print(f"{framing}\t{model}\t{alpha:g}\tINCOMPLETE")
                    continue
                median, lo, hi, p_value = result
                print(f"{framing}\t{model}\t{alpha:g}\t{len(ids)}\t"
                      f"{np.median(r_image):+.4f}\t{np.median(r_text):+.4f}\t"
                      f"{median:+.4f}\t[{lo:+.4f},{hi:+.4f}]\t{p_value:.3g}")


if __name__ == "__main__":
    main()
