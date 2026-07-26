"""Paired endpoint analysis for the same-question ChartQA conflict control."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


FILES = {
    "image": ("level_0_clean", "level_5_heavy_degradation"),
    "text": ("level_0_clean", "level_5_heavy_corruption"),
}


def load(path, mode):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if mode == "cll":
                value = (row.get("margin") or {}).get("margin_mean")
            else:
                value = {"image": 0.0, "text": 1.0}.get(row.get("follows"))
            if value is not None:
                rows[int(row["i"])] = float(value)
    return rows


def changes(root, model, arm, mode):
    l0, l5 = FILES[arm]
    before = load(root / arm / model / f"{l0}.{mode}.jsonl", mode)
    after = load(root / arm / model / f"{l5}.{mode}.jsonl", mode)
    return {item: after[item] - before[item] for item in before.keys() & after.keys()}


def paired_stats(values, seed, resamples, statistic="median"):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    try:
        p_value = float(stats.wilcoxon(values, zero_method="wilcox").pvalue)
    except ValueError:
        p_value = 1.0
    rng = np.random.default_rng(seed)
    reducer = np.mean if statistic == "mean" else np.median
    boot = np.empty(resamples)
    perm_extreme = 0
    observed = abs(float(reducer(values)))
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        boot[start:start + size] = reducer(values[indices], axis=1)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        permuted = np.abs(reducer(signs * values, axis=1))
        perm_extreme += int(np.count_nonzero(permuted >= observed))
    lo, hi = np.quantile(boot, (0.025, 0.975))
    # Add one to numerator and denominator for a valid Monte Carlo p-value.
    permutation_p = (perm_extreme + 1) / (resamples + 1)
    return float(reducer(values)), float(lo), float(hi), p_value, permutation_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/phase_control/chartqa_conflict/evidence"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--exclude-ids",
        nargs="*",
        type=int,
        default=[],
        help="Compiled item indices (the JSONL 'i' field) to exclude.",
    )
    args = parser.parse_args()
    excluded = set(args.exclude_ids)

    print(
        "mode\tmodel\tn\tR_image\tR_text\tasymmetry\t95% CI\t"
        "Wilcoxon p\tpermutation p"
    )
    for mode in ("generation", "cll"):
        for model_index, model in enumerate(args.models):
            image = changes(args.root, model, "image", mode)
            text = changes(args.root, model, "text", mode)
            ids = sorted((image.keys() & text.keys()) - excluded)
            if not ids:
                print(f"{mode}\t{model}\tINCOMPLETE")
                continue
            r_image = np.asarray([image[item] for item in ids])
            r_text = np.asarray([-text[item] for item in ids])
            contrast = r_text - r_image
            statistic = "mean" if mode == "generation" else "median"
            result = paired_stats(contrast, 20260721 + model_index, args.resamples,
                                  statistic=statistic)
            median, lo, hi, p_value, permutation_p = result
            reducer = np.mean if mode == "generation" else np.median
            print(f"{mode}\t{model}\t{len(ids)}\t{reducer(r_image):+.4f}\t"
                  f"{reducer(r_text):+.4f}\t{median:+.4f}\t"
                  f"[{lo:+.4f},{hi:+.4f}]\t{p_value:.3g}\t"
                  f"{permutation_p:.3g}")


if __name__ == "__main__":
    main()
