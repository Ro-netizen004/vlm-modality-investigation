"""Compare original and neutral-prompt CLL arm asymmetry on matched item IDs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


LEVEL_FILES = {
    "image": ("level_0_clean.cll.jsonl", "level_5_heavy_degradation.cll.jsonl"),
    "text": ("level_0_clean.cll.jsonl", "level_5_heavy_corruption.cll.jsonl"),
}


def load_margins(path: Path) -> dict[int, float]:
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                value = (row.get("margin") or {}).get("margin_mean")
                if value is not None:
                    rows[int(row["i"])] = float(value)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return rows


def arm_change(model_dir: Path, arm: str) -> dict[int, float]:
    l0_name, l5_name = LEVEL_FILES[arm]
    l0, l5 = load_margins(model_dir / l0_name), load_margins(model_dir / l5_name)
    return {item: l5[item] - l0[item] for item in sorted(l0.keys() & l5.keys())}


def asymmetry(image_dir: Path, text_dir: Path) -> dict[int, float]:
    image, text = arm_change(image_dir, "image"), arm_change(text_dir, "text")
    ids = sorted(image.keys() & text.keys())
    # Image-arm change is positive toward clean text; negate text-arm change so it
    # is positive toward the clean image. Contrast > 0 means stronger text-arm response.
    return {item: -text[item] - image[item] for item in ids}


def text_preference(path: Path):
    image = text = 0
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("follows") == "image":
                image += 1
            elif row.get("follows") == "text":
                text += 1
    return text / (text + image) if text + image else None


def behavioral_asymmetry(image_dir: Path, text_dir: Path):
    image_l0 = text_preference(image_dir / "level_0_clean.csv")
    image_l5 = text_preference(image_dir / "level_5_heavy_degradation.csv")
    text_l0 = text_preference(text_dir / "level_0_clean.csv")
    text_l5 = text_preference(text_dir / "level_5_heavy_corruption.csv")
    if None in (image_l0, image_l5, text_l0, text_l5):
        return None
    r_image = image_l5 - image_l0
    r_text = text_l0 - text_l5
    return r_image, r_text, r_text - r_image


def paired_summary(values, seed=20260721, resamples=10_000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    wilcoxon = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    rng = np.random.default_rng(seed)
    medians = np.empty(resamples)
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        medians[start:start + size] = np.median(values[indices], axis=1)
    lo, hi = np.quantile(medians, (0.025, 0.975))
    return len(values), float(np.median(values)), float(lo), float(hi), float(wilcoxon.pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--original-image-root", type=Path,
                        default=Path("results/phase6_legibility/gsm8k"))
    parser.add_argument("--original-text-root", type=Path,
                        default=Path("results/phase7_text_legibility/gsm8k"))
    parser.add_argument("--control-root", type=Path,
                        default=Path("results/phase_control/role_counterbalance"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    print("model\tn\toriginal\tneutral\tneutral-original\t95% CI\tWilcoxon p")
    for model in args.models:
        original = asymmetry(args.original_image_root / model,
                             args.original_text_root / model)
        neutral = asymmetry(args.control_root / "role_neutral" / model,
                            args.control_root / "text_legibility" / "role_neutral" / model)
        ids = sorted(original.keys() & neutral.keys())
        if not ids:
            print(f"{model}\tNO MATCHED RESULTS")
            continue
        difference = np.asarray([neutral[i] - original[i] for i in ids])
        result = paired_summary(difference, resamples=args.resamples)
        n, median_difference, lo, hi, p_value = result
        print(f"{model}\t{n}\t{np.median([original[i] for i in ids]):+.4f}\t"
              f"{np.median([neutral[i] for i in ids]):+.4f}\t{median_difference:+.4f}\t"
              f"[{lo:+.4f},{hi:+.4f}]\t{p_value:.3g}")

    print("\nRAW BEHAVIORAL ENDPOINTS (descriptive; preference among decidable trials)")
    print("model\tframing\tR_image\tR_text\tasymmetry")
    for model in args.models:
        for framing, image_dir, text_dir in (
            ("original", args.original_image_root / model,
             args.original_text_root / model),
            ("neutral", args.control_root / "role_neutral" / model,
             args.control_root / "text_legibility" / "role_neutral" / model),
        ):
            result = behavioral_asymmetry(image_dir, text_dir)
            if result is None:
                print(f"{model}\t{framing}\tINCOMPLETE")
            else:
                print(f"{model}\t{framing}\t{result[0]:+.4f}\t{result[1]:+.4f}\t"
                      f"{result[2]:+.4f}")


if __name__ == "__main__":
    main()
