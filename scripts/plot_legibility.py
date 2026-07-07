#!/usr/bin/env python3
"""
Plot the legibility curve: text preference vs. image legibility.

Reads results/phase6_legibility/legibility_all.json (or per-model
legibility_summary.json files) and draws one line per model of text preference
in the mismatch condition against noise level.

Reading of the figure:
  * FLAT curve  -> modality preference is a fixed bias, insensitive to how
                   readable the image is.
  * SLOPED curve (text preference rises as the image degrades)
                -> rational, reliability-aware arbitration: the model leans on
                   the text more as the image becomes less trustworthy.

Usage:
    python scripts/plot_legibility.py
    python scripts/plot_legibility.py --results-dir results/phase6_legibility
    python scripts/plot_legibility.py --as-percent
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.noise import NOISE_LEVELS
    LEVEL_NAMES = {L: NOISE_LEVELS[L]["name"] for L in NOISE_LEVELS}
except Exception:  # keep the plotter usable even if src.noise can't import
    LEVEL_NAMES = {}

# Distinct, print-friendly colours (same family as src/visualization.py).
PALETTE = ["#4C72B0", "#C44E52", "#55A868", "#8172B2",
           "#DD8452", "#937860", "#DA8BC3", "#8C8C8C"]

FLAT_THRESHOLD = 0.05  # spread below this reads as "flat"


def _load_summaries(results_dir):
    """Prefer the combined legibility_all.json; fall back to per-model files."""
    combined = os.path.join(results_dir, "legibility_all.json")
    if os.path.exists(combined):
        with open(combined) as f:
            data = json.load(f)
        if data:
            return data

    summaries = {}
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name, "legibility_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                summaries[name] = json.load(f)
    return summaries


def _levels_and_prefs(model_summary):
    """Return (sorted_levels, prefs) with prefs aligned to levels; None -> nan.

    JSON serialises the int level keys as strings, so normalise them back.
    """
    raw = model_summary.get("text_preference_by_level", {})
    pref_by_level = {int(k): v for k, v in raw.items()}
    levels = sorted(pref_by_level)
    prefs = [pref_by_level[L] if pref_by_level[L] is not None else np.nan
             for L in levels]
    return levels, prefs


def plot(results_dir, as_percent=False, out_path=None):
    summaries = _load_summaries(results_dir)
    if not summaries:
        print(f"No legibility results found in {results_dir} "
              f"(expected legibility_all.json or <model>/legibility_summary.json)")
        return None

    scale = 100.0 if as_percent else 1.0
    ylabel = "Text preference (%)" if as_percent else "Text preference"

    # Union of all levels present, in increasing-degradation order.
    all_levels = sorted({L for s in summaries.values()
                         for L in _levels_and_prefs(s)[0]})
    x_index = {L: i for i, L in enumerate(all_levels)}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    spreads = {}
    for idx, (model, summary) in enumerate(sorted(summaries.items())):
        levels, prefs = _levels_and_prefs(summary)
        if not levels:
            continue
        xs = [x_index[L] for L in levels]
        ys = [p * scale for p in prefs]
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=2,
                color=color, label=model)

        valid = [p for p in prefs if not np.isnan(p)]
        if valid:
            spreads[model] = max(valid) - min(valid)

    # X ticks: level number + human-readable corruption name.
    tick_labels = []
    for L in all_levels:
        name = LEVEL_NAMES.get(L, "")
        tick_labels.append(f"L{L}\n{name}" if name else f"L{L}")
    ax.set_xticks(range(len(all_levels)))
    ax.set_xticklabels(tick_labels, fontsize=9)

    ax.set_xlabel("Image corruption level  (less legible →)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 105 if as_percent else 1.05)
    ax.grid(axis="both", alpha=0.3)
    ax.legend(fontsize=10, title="Model", loc="lower right")

    # Interpretation subtitle from the largest per-model spread.
    if spreads:
        widest_model = max(spreads, key=spreads.get)
        widest = spreads[widest_model]
        if widest < FLAT_THRESHOLD:
            verdict = (f"Flat across models (max spread {widest*100:.1f}pp) "
                       f"→ fixed text bias, insensitive to legibility")
        else:
            verdict = (f"{widest_model} shifts {widest*100:.1f}pp "
                       f"→ preference tracks legibility (rational arbitration)")
    else:
        verdict = ""

    ax.set_title("Modality preference under image degradation",
                 fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    if verdict:  # interpretation caption below the x-axis label (kept clear of the title)
        fig.text(0.5, -0.02, verdict, ha="center", va="top",
                 fontsize=10, style="italic", color="#444444")
    if out_path is None:
        out_path = os.path.join(results_dir, "text_preference_vs_legibility.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Console summary for quick reading without opening the PNG.
    print("\nText preference by level:")
    for model, summary in sorted(summaries.items()):
        levels, prefs = _levels_and_prefs(summary)
        cells = " ".join(
            f"L{L}={p:.3f}" if not np.isnan(p) else f"L{L}=--"
            for L, p in zip(levels, prefs))
        sp = spreads.get(model)
        tail = f"  (spread {sp*100:.1f}pp)" if sp is not None else ""
        print(f"  {model:30s} {cells}{tail}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot text preference vs. image legibility")
    parser.add_argument("--results-dir", default="results/phase6_legibility")
    parser.add_argument("--as-percent", action="store_true",
                        help="Plot preference as a percentage instead of a 0-1 fraction.")
    parser.add_argument("--out", default=None, help="Output PNG path (optional).")
    args = parser.parse_args()
    plot(args.results_dir, as_percent=args.as_percent, out_path=args.out)


if __name__ == "__main__":
    main()
