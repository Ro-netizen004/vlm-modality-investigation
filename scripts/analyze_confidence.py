"""Confidence-trajectory analysis for blackbox/API models (OpenAI/Gemini).

The clean text-vs-image CLL margin isn't available for API models (no continuation
scoring). But the logprob of the model's OWN generated answer IS available and
uncensored. This script extracts that answer-token logprob per trial from the
`level_X.logprobs.jsonl` sidecars and reports how confidence changes with legibility.

Question: does the model's certainty in its (text) answer RISE as the image degrades,
even after the binary preference saturates? That's the frontier ceiling-cracker on the
confidence axis.

Usage:
    python scripts/analyze_confidence.py results/phase6_legibility/GPT-5.6-Luna
    python scripts/analyze_confidence.py <model_dir> --plot
"""
import argparse, csv, json, os, statistics as st

LEVELS = [(0, "level_0_clean"), (2, "level_2_blur_light"),
          (4, "level_4_blur_noise"), (5, "level_5_heavy_degradation")]


def answer_confidence(logprobs):
    """Mean logprob of the numeric answer tokens (those after the '####' marker)."""
    if not logprobs:
        return None
    toks = [e.get("tok", "") for e in logprobs]
    text = "".join(toks)
    p = text.rfind("####")
    if p < 0:
        return None
    p += 4  # char position just after '####'
    pos, ans = 0, []
    for e in logprobs:
        if pos >= p and any(c.isdigit() for c in e.get("tok", "")):
            ans.append(e["lp"])
        elif pos >= p and ans:
            break  # answer number ended
        pos += len(e.get("tok", ""))
    return sum(ans) / len(ans) if ans else None


def load_level(model_dir, name):
    """Join the logprobs sidecar with the generation CSV on problem index."""
    lp_path = os.path.join(model_dir, f"{name}.logprobs.jsonl")
    csv_path = os.path.join(model_dir, f"{name}.csv")
    if not os.path.exists(lp_path):
        return []
    follows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                follows[int(row["problem_id"])] = row.get("follows")
    out = []
    with open(lp_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            conf = answer_confidence(r.get("logprobs"))
            if conf is not None:
                out.append({"i": r["i"], "conf": conf, "follows": follows.get(r["i"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="dir with level_*.logprobs.jsonl + level_*.csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print(f"Answer-confidence trajectory: {os.path.basename(args.model_dir.rstrip('/'))}")
    print(f"  (mean logprob of the generated answer tokens; higher = more certain)\n")
    print(f"  {'lvl':>4}{'n':>7}{'mean_conf':>11}{'median':>10}{'text_only_mean':>16}")
    rows_for_plot = []
    for L, name in LEVELS:
        rows = load_level(args.model_dir, name)
        if not rows:
            print(f"  {L:>4}   (no logprobs sidecar — API run may predate logprob capture)")
            continue
        confs = [r["conf"] for r in rows]
        txt = [r["conf"] for r in rows if r["follows"] == "text"]
        tmean = st.mean(txt) if txt else float("nan")
        print(f"  {L:>4}{len(rows):>7}{st.mean(confs):>11.3f}{st.median(confs):>10.3f}{tmean:>16.3f}")
        rows_for_plot.append((L, st.mean(confs), tmean))

    if args.plot and rows_for_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r[0] for r in rows_for_plot]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(xs, [r[1] for r in rows_for_plot], "o-", label="all trials")
        ax.plot(xs, [r[2] for r in rows_for_plot], "s--", label="text-followed only", alpha=0.7)
        ax.set_xlabel("noise level (0 clean → 5 heavy)")
        ax.set_ylabel("mean answer-token logprob (confidence)")
        ax.set_title(f"Answer confidence vs. image legibility\n{os.path.basename(args.model_dir.rstrip('/'))}")
        ax.set_xticks(xs); ax.grid(alpha=0.25); ax.legend(fontsize=9)
        fig.tight_layout()
        out = os.path.join(args.model_dir, "confidence_vs_legibility.png")
        fig.savefig(out, dpi=140)
        print(f"\nsaved {out}")
        print("Rising = certainty in the text answer grows as the image degrades")
        print("        (frontier ceiling-cracker on the confidence axis).")


if __name__ == "__main__":
    main()
