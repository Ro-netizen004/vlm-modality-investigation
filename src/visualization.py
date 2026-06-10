"""Plotting and visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter


def plot_accuracy_comparison(results_by_model, output_dir):
    """Bar chart comparing text-only vs rendered image accuracy across models."""
    models = list(results_by_model.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_models), 6))

    x = np.arange(n_models)
    width = 0.35

    text_accs = [results_by_model[m]["acc_text"] * 100 for m in models]
    img_accs = [results_by_model[m]["acc_img"] * 100 for m in models]

    text_cis = [results_by_model[m].get("text_ci_95", (0, 0)) for m in models]
    img_cis = [results_by_model[m].get("img_ci_95", (0, 0)) for m in models]

    text_errs = [[a - ci[0] * 100, ci[1] * 100 - a] for a, ci in zip(text_accs, text_cis)]
    img_errs = [[a - ci[0] * 100, ci[1] * 100 - a] for a, ci in zip(img_accs, img_cis)]

    bars1 = ax.bar(x - width / 2, text_accs, width, label="Text-Only",
                   color="#4C72B0", edgecolor="black",
                   yerr=np.array(text_errs).T, capsize=4)
    bars2 = ax.bar(x + width / 2, img_accs, width, label="Rendered Image",
                   color="#55A868", edgecolor="black",
                   yerr=np.array(img_errs).T, capsize=4)

    for bar, acc in zip(list(bars1) + list(bars2), text_accs + img_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    short_names = [m.split("/")[-1] for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("GSM8K Zero-Shot Accuracy by Input Modality", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_breakdown(errors_by_condition, model_name, output_dir):
    """Grouped bar chart of error types per condition."""
    categories = ["correct", "arithmetic_error", "reasoning_error", "no_number", "vision_error"]
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    cond_labels = list(errors_by_condition.keys())
    n_conds = len(cond_labels)

    x = np.arange(len(categories))
    bar_w = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, label in enumerate(cond_labels):
        c = Counter(errors_by_condition[label])
        counts = [c.get(cat, 0) for cat in categories]
        offset = (idx - n_conds / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, counts, bar_w, label=label,
                      color=palette[idx % len(palette)], edgecolor="black", alpha=0.85)
        for b, v in zip(bars, counts):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                        str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=11)
    ax.set_ylabel("Number of Problems", fontsize=12)
    short = model_name.split("/")[-1]
    ax.set_title(f"Error Type Breakdown — {short}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"error_breakdown_{short}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_mismatch_dominance(follow_counts, model_name, output_dir):
    """Bar chart of modality dominance in mismatch condition."""
    keys = ["image", "text", "equal", "invalid"]
    labels = ["Closer to\nImage", "Closer to\nText", "Equal\nDistance", "Invalid\n(no number)"]
    counts = [follow_counts.get(k, 0) for k in keys]
    colors = ["#55A868", "#4C72B0", "#8172B2", "#C44E52"]
    n_total = sum(counts)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", width=0.5)
    for bar, v in zip(bars, counts):
        if v > 0:
            pct = v / n_total * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v}\n({pct:.0f}%)", ha="center", fontsize=11, fontweight="bold")

    short = model_name.split("/")[-1]
    ax.set_ylabel("Number of Problems")
    ax.set_title(f"Modality Dominance Under Conflict — {short}")
    ax.set_ylim(0, n_total + max(15, n_total * 0.15))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"mismatch_dominance_{short}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_statistical_summary(all_stats, output_dir):
    """Summary plot with CIs, effect sizes, and p-values across all models."""
    models = list(all_stats.keys())
    n = len(models)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Accuracy with CIs
    ax = axes[0]
    for i, m in enumerate(models):
        s = all_stats[m]
        short = m.split("/")[-1]
        ax.errorbar(i - 0.1, s["acc_text"] * 100,
                     yerr=[[s["acc_text"] * 100 - s["text_ci_95"][0] * 100],
                           [s["text_ci_95"][1] * 100 - s["acc_text"] * 100]],
                     fmt="o", color="#4C72B0", capsize=5, markersize=8, label="Text" if i == 0 else "")
        ax.errorbar(i + 0.1, s["acc_img"] * 100,
                     yerr=[[s["acc_img"] * 100 - s["img_ci_95"][0] * 100],
                           [s["img_ci_95"][1] * 100 - s["acc_img"] * 100]],
                     fmt="s", color="#55A868", capsize=5, markersize=8, label="Image" if i == 0 else "")

    ax.set_xticks(range(n))
    ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy with 95% CI")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Effect sizes
    ax = axes[1]
    effects = [all_stats[m]["cohens_h"] for m in models]
    colors = ["#C44E52" if e < -0.5 else "#DDAA33" if e < -0.2 else "#55A868" for e in effects]
    ax.barh(range(n), effects, color=colors, edgecolor="black")
    ax.set_yticks(range(n))
    ax.set_yticklabels([m.split("/")[-1] for m in models], fontsize=9)
    ax.set_xlabel("Cohen's h")
    ax.set_title("Effect Size (Text vs Image)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=-0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="x", alpha=0.3)

    # Panel 3: McNemar p-values
    ax = axes[2]
    pvals = [all_stats[m]["mcnemar_p"] for m in models]
    log_pvals = [-np.log10(max(p, 1e-20)) for p in pvals]
    colors = ["#C44E52" if p < 0.05 else "#AAAAAA" for p in pvals]
    ax.barh(range(n), log_pvals, color=colors, edgecolor="black")
    ax.set_yticks(range(n))
    ax.set_yticklabels([m.split("/")[-1] for m in models], fontsize=9)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("McNemar's Test Significance")
    ax.axvline(x=-np.log10(0.05), color="red", linewidth=1, linestyle="--", label="p=0.05")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Statistical Summary Across Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "statistical_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
