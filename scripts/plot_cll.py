"""Plot the CLL arbitration margin vs legibility level, per model.
Left: raw median margin (absolute text-over-image preference strength).
Right: change from clean baseline (the ceiling-cracker trajectory) — statistically
significant risers highlighted."""
import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "cll_gaivi")
LEVELS = [0, 2, 4, 5]
LEVEL_FILE = {0: "level_0_clean", 2: "level_2_blur_light",
              4: "level_4_blur_noise", 5: "level_5_heavy_degradation"}
# From analyze_cll.py section 5 (MWU + Spearman): significant graded risers.
RISERS = {"Qwen2.5-VL-7B-Instruct", "Idefics3-8B-Llama3"}


def median_margins(model):
    out = []
    for L in LEVELS:
        p = os.path.join(ROOT, model, f"{LEVEL_FILE[L]}.cll.jsonl")
        vals = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("margin"):
                    vals.append(r["margin"]["margin_mean"])
        out.append(st.median(vals))
    return out


models = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
data = {m: median_margins(m) for m in models}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
for m in models:
    ys = data[m]
    riser = m in RISERS
    style = dict(marker="o", linewidth=2.4 if riser else 1.4,
                 linestyle="-" if riser else "--",
                 alpha=1.0 if riser else 0.55,
                 zorder=3 if riser else 2)
    label = m + ("  ★" if riser else "")
    axL.plot(LEVELS, ys, label=label, **style)
    axR.plot(LEVELS, [y - ys[0] for y in ys], label=label, **style)

axL.set_title("Raw CLL margin  (median CLL$_{text}$ − CLL$_{image}$)")
axL.set_ylabel("median margin (nats)")
axR.set_title("Shift from clean baseline  (ceiling-cracker)")
axR.set_ylabel("Δ median margin vs level 0 (nats)")
axR.axhline(0, color="gray", lw=0.8, ls=":")
for ax in (axL, axR):
    ax.set_xlabel("noise level (0 clean → 5 heavy)")
    ax.set_xticks(LEVELS)
    ax.grid(alpha=0.25)
axR.legend(fontsize=8, title="★ = significant graded rise")
fig.suptitle("Text-over-image answer preference in probability space vs. image legibility\n"
             "(argmax preference is pinned ~99% for all; the graded margin reveals hidden reliability-sensitivity)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(ROOT, "margin_vs_legibility.png")
fig.savefig(out, dpi=140)
print("saved", out)
