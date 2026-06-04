"""Result visualization utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(cfg: dict, df: pd.DataFrame, summary: dict | None = None) -> None:
    plot_dir = Path(cfg["output_dir"])
    run_name = cfg["run_name"]
    mode = cfg["experiment_mode"]
    model_tag = cfg["model_id"].split("/")[-1]

    accuracy = df["correct"].mean() if len(df) else 0.0
    error_counts = Counter(df["error_type"])

    # Accuracy bar
    fig, ax = plt.subplots(figsize=(5, 5))
    bar = ax.bar([mode], [accuracy * 100], color="#4C72B0", width=0.4, edgecolor="black")
    ax.text(
        bar[0].get_x() + bar[0].get_width() / 2,
        bar[0].get_height() + 1,
        f"{accuracy:.1%}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"GSM8K Accuracy\n{model_tag}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = plot_dir / f"{run_name}_accuracy.png"
    plt.savefig(p, dpi=150)
    plt.close()

    # Error breakdown
    error_cats = ["correct", "arithmetic_error", "reasoning_error", "no_number", "vision_error"]
    counts_plot = [error_counts.get(c, 0) for c in error_cats]
    colors_plot = ["#55A868", "#C44E52", "#DD8452", "#8172B2", "#937860"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(error_cats, counts_plot, color=colors_plot, edgecolor="black", width=0.5)
    for b, v in zip(bars, counts_plot):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(v), ha="center", va="bottom")
    ax.set_xticklabels([c.replace("_", "\n") for c in error_cats])
    ax.set_ylabel("Number of Problems")
    ax.set_title(f"Error Type Breakdown — {mode}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = plot_dir / f"{run_name}_error_breakdown.png"
    plt.savefig(p, dpi=150)
    plt.close()

    # Mismatch dominance
    if mode == "mismatch" and "mismatch_follows" in df.columns:
        follow_counts = Counter(df["mismatch_follows"])
        labels_mm = ["Follows Image", "Follows Text", "Equal", "Invalid"]
        keys_mm = ["image", "text", "equal", "invalid"]
        vals_mm = [follow_counts.get(k, 0) for k in keys_mm]
        colors_mm = ["#55A868", "#4C72B0", "#8172B2", "#999999"]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels_mm, vals_mm, color=colors_mm, edgecolor="black", width=0.5)
        for b, v in zip(bars, vals_mm):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(v), ha="center", va="bottom")
        ax.set_ylabel("Number of Problems")
        ax.set_title(f"Mismatch Dominance\n{model_tag}")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = plot_dir / f"{run_name}_mismatch_dominance.png"
        plt.savefig(p, dpi=150)
        plt.close()
