"""Plot GSM8K CLL trajectories for both degradation arms with bootstrap uncertainty.

Raw panels show level-wise medians with bootstrap 95% CIs. Shift panels preserve
the repeated-item design: each level is joined to L0 by item id, and confidence
intervals bootstrap the within-item differences. The output is Figure 1 in the paper.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/figures/margin_vs_legibility_both.png"
LEVELS = (0, 2, 4, 5)
FILES = {
    "image": {
        0: "level_0_clean.cll.jsonl",
        2: "level_2_blur_light.cll.jsonl",
        4: "level_4_blur_noise.cll.jsonl",
        5: "level_5_heavy_degradation.cll.jsonl",
    },
    "text": {
        0: "level_0_clean.cll.jsonl",
        2: "level_2_light_corruption.cll.jsonl",
        4: "level_4_medium_corruption.cll.jsonl",
        5: "level_5_heavy_corruption.cll.jsonl",
    },
}
DISPLAY = {
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen2-VL-2B-Instruct": "Qwen2-VL-2B",
    "Idefics3-8B-Llama3": "Idefics3-8B",
    "Phi-3.5-vision-instruct": "Phi-3.5",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-1.6-7B",
}


def load_level(model, arm, level):
    phase = "phase6_legibility" if arm == "image" else "phase7_text_legibility"
    path = ROOT / "results" / phase / "gsm8k" / model / FILES[arm][level]
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


def bootstrap_median(values, rng, n_resamples=10_000, batch=500):
    values = np.asarray(values, dtype=float)
    estimate = float(np.median(values))
    boot = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        boot[start:start + size] = np.median(values[idx], axis=1)
    lo, hi = np.quantile(boot, (0.025, 0.975))
    return estimate, float(lo), float(hi)


def trajectory(model, arm, rng):
    per = {level: load_level(model, arm, level) for level in LEVELS}
    if any(not per[level] for level in LEVELS):
        return None
    raw, shift = [], []
    for level in LEVELS:
        vals = list(per[level].values())
        raw.append(bootstrap_median(vals, rng))
        ids = sorted(per[0].keys() & per[level].keys())
        diffs = [per[level][i] - per[0][i] for i in ids]
        shift.append(bootstrap_median(diffs, rng))
    return raw, shift


def draw_panel(ax, series, title, ylabel):
    colors = plt.get_cmap("tab10")
    for index, (model, points) in enumerate(series.items()):
        center = np.asarray([p[0] for p in points])
        lower = center - np.asarray([p[1] for p in points])
        upper = np.asarray([p[2] for p in points]) - center
        ax.errorbar(
            LEVELS, center, yerr=np.vstack([lower, upper]), label=DISPLAY[model],
            color=colors(index), marker="o", markersize=4, linewidth=1.7,
            elinewidth=1.0, capsize=2.5, alpha=0.9,
        )
    ax.axhline(0, color="0.45", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("degradation level (0 = clean)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(LEVELS)
    ax.grid(alpha=0.2)


def main():
    rng = np.random.default_rng(20260720)
    data = {"image": {}, "text": {}}
    for arm in data:
        for model in DISPLAY:
            result = trajectory(model, arm, rng)
            if result is not None:
                data[arm][model] = result

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    draw_panel(axes[0, 0], {m: v[0] for m, v in data["image"].items()},
               "Image degraded: raw margin", "median CLL margin (nats/token)")
    draw_panel(axes[0, 1], {m: v[1] for m, v in data["image"].items()},
               "Image degraded: paired shift from L0", "median paired shift (nats/token)")
    draw_panel(axes[1, 0], {m: v[0] for m, v in data["text"].items()},
               "Text degraded: raw margin", "median CLL margin (nats/token)")
    draw_panel(axes[1, 1], {m: v[1] for m, v in data["text"].items()},
               "Text degraded: paired shift from L0", "median paired shift (nats/token)")
    axes[0, 1].legend(fontsize=7.5, ncol=2, frameon=False, loc="best")
    axes[1, 1].legend(fontsize=7.5, ncol=2, frameon=False, loc="best")
    fig.suptitle("CLL arbitration margin under image vs. text degradation (GSM8K)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
