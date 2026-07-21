"""Paired L0->L5 CLL analysis for both degradation arms.

Rows are joined by the stable item id ``i``.  The primary test is Wilcoxon's
signed-rank test on within-item margin changes.  We also report a Monte-Carlo
paired permutation p-value (random sign flips) and a paired bootstrap confidence
interval for the median within-item change.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LEVEL_FILES = {
    "image": ("level_0_clean.cll.jsonl", "level_5_heavy_degradation.cll.jsonl"),
    "text": ("level_0_clean.cll.jsonl", "level_5_heavy_corruption.cll.jsonl"),
}
DISPLAY = {
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen2-VL-2B-Instruct": "Qwen2-VL-2B",
    "Idefics3-8B-Llama3": "Idefics3-8B",
    "Phi-3.5-vision-instruct": "Phi-3.5",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-1.6-7B",
}


def _load_by_id(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            margin = (row.get("margin") or {}).get("margin_mean")
            if margin is not None:
                rows[int(row["i"])] = float(margin)
    return rows


def paired_differences_by_id(model, benchmark, arm):
    base = ROOT / ("results/phase6_legibility" if arm == "image" else
                   "results/phase7_text_legibility") / benchmark / model
    l0 = _load_by_id(base / LEVEL_FILES[arm][0])
    l5 = _load_by_id(base / LEVEL_FILES[arm][1])
    # Raw margin change: positive means toward text, negative means toward image.
    return {i: l5[i] - l0[i] for i in sorted(l0.keys() & l5.keys())}


def paired_differences(model, benchmark, arm):
    return np.asarray(list(paired_differences_by_id(model, benchmark, arm).values()),
                      dtype=float)


def paired_stats(differences, seed=20260720, n_resamples=10_000):
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    wilcoxon = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox",
                              method="auto")
    rng = np.random.default_rng(seed)

    # Paired randomization test: under the null, each within-item difference may
    # have either sign.  Use the mean as the randomization statistic.
    observed = abs(float(np.mean(d)))
    extreme = 0
    batch = 2_000
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(d)))
        extreme += int(np.count_nonzero(np.abs(np.mean(signs * d, axis=1)) >= observed))
    permutation_p = (extreme + 1) / (n_resamples + 1)

    # Paired bootstrap: resample item-level differences, not the two levels
    # independently.  The estimand is the median within-item change.
    medians = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        sample_idx = rng.integers(0, len(d), size=(size, len(d)))
        medians[start:start + size] = np.median(d[sample_idx], axis=1)
    lo, hi = np.quantile(medians, (0.025, 0.975))
    return {
        "n": len(d),
        "median_paired_change": float(np.median(d)),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "permutation_p": float(permutation_p),
        "bootstrap_ci": (float(lo), float(hi)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    for arm in ("image", "text"):
        print(f"\n{arm.upper()} ARM")
        for benchmark in ("gsm8k", "svamp"):
            print(f"  {benchmark}")
            for model in DISPLAY:
                result = paired_stats(paired_differences(model, benchmark, arm),
                                      n_resamples=args.resamples)
                if result is None:
                    continue
                lo, hi = result["bootstrap_ci"]
                print(f"    {DISPLAY[model]:17s} n={result['n']:4d} "
                      f"median_d={result['median_paired_change']:+.3f} "
                      f"W_p={result['wilcoxon_p']:.3g} "
                      f"perm_p={result['permutation_p']:.3g} "
                      f"boot95=[{lo:+.3f}, {hi:+.3f}]")

    print("\nPAIRED ARM CONTRAST (R_text - R_image; positive = larger text-arm reallocation)")
    latex_rows = []
    for benchmark in ("gsm8k", "svamp"):
        print(f"  {benchmark}")
        for model in DISPLAY:
            image_change = paired_differences_by_id(model, benchmark, "image")
            text_change = paired_differences_by_id(model, benchmark, "text")
            ids = sorted(image_change.keys() & text_change.keys())
            if not ids:
                continue
            r_image = np.asarray([image_change[i] for i in ids])
            r_text = np.asarray([-text_change[i] for i in ids])
            contrast = r_text - r_image
            result = paired_stats(contrast, seed=20260721,
                                  n_resamples=args.resamples)
            lo, hi = result["bootstrap_ci"]
            print(f"    {DISPLAY[model]:17s} n={result['n']:4d} "
                  f"median_contrast={result['median_paired_change']:+.3f} "
                  f"W_p={result['wilcoxon_p']:.3g} "
                  f"perm_p={result['permutation_p']:.3g} "
                  f"boot95=[{lo:+.3f}, {hi:+.3f}]")
            latex_rows.append((benchmark, DISPLAY[model], np.median(r_image),
                               np.median(r_text), result))

    print("\n% --- Direct paired asymmetry table ---")
    print("\\begin{table*}[t]\\centering\\small")
    print("\\begin{tabular}{llrrrrr}")
    print("\\toprule")
    print("Benchmark & Model & $R_I$ & $R_T$ & $R_T-R_I$ & 95\\% CI & $p_W$ \\\\")
    print("\\midrule")
    previous = None
    for benchmark, model, r_image, r_text, result in latex_rows:
        if previous is not None and benchmark != previous:
            print("\\midrule")
        bench = benchmark.upper() if benchmark == "svamp" else "GSM8K"
        lo, hi = result["bootstrap_ci"]
        print(f"{bench} & {model} & {r_image:+.2f} & {r_text:+.2f} & "
              f"{result['median_paired_change']:+.2f} & "
              f"$[{lo:+.2f},{hi:+.2f}]$ & {result['wilcoxon_p']:.1e} \\\\")
        previous = benchmark
    print("\\bottomrule\n\\end{tabular}")
    print("\\caption{Direct paired asymmetry contrast. $R_I$ and $R_T$ are median "
          "within-item reallocations toward the clean channel. The contrast is computed "
          "within item; intervals are paired-bootstrap 95\\% CIs and $p_W$ is the "
          "two-sided Wilcoxon signed-rank p-value.}")
    print("\\label{tab:direct_asymmetry}\n\\end{table*}")


if __name__ == "__main__":
    main()
