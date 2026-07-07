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


def _load_attention(attention_dir):
    """Load the Phase 7 attention summaries (attention_all.json or per-model)."""
    if not attention_dir or not os.path.isdir(attention_dir):
        return {}
    combined = os.path.join(attention_dir, "attention_all.json")
    if os.path.exists(combined):
        with open(combined) as f:
            data = json.load(f)
        if data:
            return data
    out = {}
    for name in sorted(os.listdir(attention_dir)):
        path = os.path.join(attention_dir, name, "attention_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


def _levels_and_attn(model_summary):
    """Return (sorted_levels, attn) from a Phase 7 summary; None -> nan."""
    raw = model_summary.get("attention_by_level", {})
    by_level = {int(k): v for k, v in raw.items()}
    levels = sorted(by_level)
    attn = [by_level[L] if by_level[L] is not None else np.nan for L in levels]
    return levels, attn


def plot(results_dir, as_percent=False, out_path=None, benchmark="gsm8k",
         attention_dir=None):
    summaries = _load_summaries(results_dir)
    if not summaries:
        print(f"No legibility results found in {results_dir} "
              f"(expected legibility_all.json or <model>/legibility_summary.json)")
        return None

    attn_summaries = _load_attention(attention_dir)

    scale = 100.0 if as_percent else 1.0
    ylabel = "Text preference (%)" if as_percent else "Text preference"

    # Union of all levels present, in increasing-degradation order.
    all_levels = sorted({L for s in summaries.values()
                         for L in _levels_and_prefs(s)[0]})
    x_index = {L: i for i, L in enumerate(all_levels)}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    spreads = {}
    color_of = {}
    for idx, (model, summary) in enumerate(sorted(summaries.items())):
        levels, prefs = _levels_and_prefs(summary)
        if not levels:
            continue
        xs = [x_index[L] for L in levels]
        ys = [p * scale for p in prefs]
        color = PALETTE[idx % len(PALETTE)]
        color_of[model] = color
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=2,
                color=color, label=model)

        valid = [p for p in prefs if not np.isnan(p)]
        if valid:
            spreads[model] = max(valid) - min(valid)

    # ── Phase 7 overlay: text->image attention on a secondary axis (dashed) ──
    ax2 = None
    if attn_summaries:
        ax2 = ax.twinx()
        for j, (model, summ) in enumerate(sorted(attn_summaries.items())):
            levels, attn = _levels_and_attn(summ)
            if not levels:
                continue
            xs = [x_index[L] for L in levels if L in x_index]
            ys = [a for L, a in zip(levels, attn) if L in x_index]
            color = color_of.get(model, PALETTE[(len(color_of) + j) % len(PALETTE)])
            ax2.plot(xs, ys, marker="s", markersize=6, linewidth=1.8,
                     linestyle="--", color=color, alpha=0.9)
        ax2.set_ylabel("Text→image attention  (dashed)", fontsize=11, color="#555555")
        ax2.tick_params(axis="y", labelcolor="#555555")
        ax2.set_ylim(bottom=0)

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
    legend_title = "Model (solid=pref, dashed=attn)" if attn_summaries else "Model"
    ax.legend(fontsize=10, title=legend_title, loc="lower right")

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

    ax.set_title(f"Modality preference under image degradation — {benchmark.upper()}",
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
    parser.add_argument("--benchmark", default="gsm8k",
                        help="Which benchmark's results to plot. gsm8k reads --results-dir "
                             "directly; others read <results-dir>/<benchmark>/.")
    parser.add_argument("--as-percent", action="store_true",
                        help="Plot preference as a percentage instead of a 0-1 fraction.")
    parser.add_argument("--attention-results-dir", default=None,
                        help="Optional Phase 7 attention results dir (results/phase7_attention). "
                             "If given, overlays text→image attention (dashed, right axis).")
    parser.add_argument("--out", default=None, help="Output PNG path (optional).")
    args = parser.parse_args()

    # Match run_legibility's namespacing: gsm8k at the root, others in a subdir.
    results_dir = (args.results_dir if args.benchmark == "gsm8k"
                   else os.path.join(args.results_dir, args.benchmark))
    attention_dir = None
    if args.attention_results_dir:
        attention_dir = (args.attention_results_dir if args.benchmark == "gsm8k"
                         else os.path.join(args.attention_results_dir, args.benchmark))
    plot(results_dir, as_percent=args.as_percent, out_path=args.out,
         benchmark=args.benchmark, attention_dir=attention_dir)


if __name__ == "__main__":
    main()
