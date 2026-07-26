#!/usr/bin/env python3
"""Confound-controlled legibility analysis: after accounting for how much information each
channel actually loses, is there still a modality (text-vs-image) arbitration effect?

Triangulates TWO independent legibility axes:
  * task_acc   : single-modality VLM accuracy   (measure_legibility_decodability.py) -- what the model can EXPLOIT
  * survival   : OCR / char survival            (measure_legibility_survival.py)     -- PERCEPTUAL information available

...against the arbitration shift from the mismatch runs:
  * --metric cll (default): median CLL margin per level   |  behavioral: text_preference_by_level

For each legibility axis it fits the interaction model over points (channel x level x model):
    arb_shift ~ b0 + b1*leg_loss + b2*modality + b3*(modality x leg_loss)
      leg_loss = fractional legibility loss of the DEGRADED channel = (L0 - L)/L0   (0..1)
      modality = 0 image, 1 text ; arb_shift = shift toward the CLEAN channel (>=0 = reliability-aware)
      b1        = image-channel response per unit legibility loss
      b1+b3     = text-channel response ;  b3 = the KEY interaction (text slope - image slope)
    b3 ~ 0 (n.s.) -> asymmetry explained by legibility loss (rational, symmetric weighting).
    b3 > 0 (sig.) -> residual text-primacy beyond legibility loss (asymmetry survives the control).

Reporting b3 consistently across BOTH legibility axes = the strong robustness statement.

Usage:
    python scripts/analyze_legibility_control.py --benchmark svamp
"""
import argparse
import json
import os
import re
import sys
import statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LEVELS = [0, 2, 4, 5]
CLL_RE = re.compile(r"^level_(\d+)_.*\.cll\.jsonl$")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _intkeys(raw):
    out = {}
    for k, v in (raw or {}).items():
        try:
            out[int(k)] = v
        except (ValueError, TypeError):
            pass
    return out


def median_margins(model_dir):
    per = {}
    if not os.path.isdir(model_dir):
        return per
    for fn in os.listdir(model_dir):
        m = CLL_RE.match(fn)
        if not m:
            continue
        L = int(m.group(1)); vals = []
        with open(os.path.join(model_dir, fn)) as f:
            for line in f:
                try:
                    mm = (json.loads(line).get("margin") or {}).get("margin_mean")
                    if mm is not None:
                        vals.append(mm)
                except json.JSONDecodeError:
                    pass
        if vals:
            per[L] = st.median(vals)
    return per


def ols_interaction(points):
    """points: [(leg_loss, arb_shift, modality)]  modality 0=image 1=text.
    Fit arb ~ b0 + b1*leg + b2*mod + b3*(mod*leg); return coefs + p-values (or None)."""
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return None
    pts = [(x, y, m) for x, y, m in points if x is not None and y is not None]
    if len(pts) < 5:
        return None
    X = np.array([[1.0, x, m, m * x] for x, y, m in pts])
    Y = np.array([y for x, y, m in pts])
    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) < k:
        return None
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.clip(np.diag(sigma2 * XtX_inv), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), dof)
    names = ["intercept", "leg_loss(b1)", "modality(b2)", "mod:leg(b3)"]
    return dict(names=names, beta=beta.tolist(), se=se.tolist(),
                t=t.tolist(), p=p.tolist(), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="svamp")
    ap.add_argument("--metric", choices=["cll", "behavioral"], default="cll")
    ap.add_argument("--decodability", default=None)
    ap.add_argument("--survival", default=None)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--text-root", default=None)
    ap.add_argument(
        "--phase4-image-root",
        default=None,
        help=(
            "Optional fallback directory containing Phase 4 per-model "
            "level_<L>_*.json image-only accuracy files. Used only when the "
            "decodability JSON has no image value for a requested level."
        ),
    )
    ap.add_argument("--min-headroom", type=float, default=0.30,
                    help="task_acc axis: drop a channel if its L0 accuracy < this")
    args = ap.parse_args()
    bm = args.benchmark

    decod = _load(args.decodability or f"results/phase_control/decodability/{bm}/decodability_all.json")
    surv = _load(args.survival or f"results/phase_control/survival/{bm}/survival.json")
    if decod is None and surv is None:
        raise SystemExit(f"need at least one legibility axis for {bm}: run "
                         f"measure_legibility_decodability.py and/or measure_legibility_survival.py")
    img_root = args.image_root or f"results/phase6_legibility/{bm}"
    txt_root = args.text_root or f"results/phase7_text_legibility/{bm}"

    def arb(model, root):
        if args.metric == "cll":
            return median_margins(os.path.join(root, model))
        summ = _load(os.path.join(root, model, "legibility_summary.json")) \
            or (_load(os.path.join(root, "legibility_all.json")) or {}).get(model)
        return _intkeys((summ or {}).get("text_preference_by_level"))

    # arbitration shift toward the clean channel, per (model, channel, level)
    shifts = {}  # (model, 'image'|'text', L) -> y
    models = set()
    root_for = {"image": img_root, "text": txt_root}
    # discover models from whichever legibility source we have
    cand = set((decod or {}).keys()) if decod else set()
    if not cand and surv is not None:
        # survival is model-independent; use models that have arbitration on both arms
        for m in os.listdir(txt_root) if os.path.isdir(txt_root) else []:
            if arb(m, img_root) and arb(m, txt_root):
                cand.add(m)
    for m in sorted(cand):
        ia, ta = arb(m, img_root), arb(m, txt_root)
        if not (ia and ta and 0 in ia and 0 in ta):
            continue
        models.add(m)
        for L in LEVELS:
            if L in ia:
                shifts[(m, "image", L)] = ia[L] - ia[0]      # image degraded -> shift toward text
            if L in ta:
                shifts[(m, "text", L)] = ta[0] - ta[L]        # text degraded  -> shift toward image
    if not models:
        raise SystemExit(f"no models have {args.metric} arbitration on both arms for {bm}.")
    print(f"\n=== Confound-controlled legibility ({bm}, arbitration={args.metric}) ===")
    print(f"models: {sorted(models)}\n")

    # ── legibility-loss providers ──
    def taskacc_loss(m, ch, L):
        acc = _intkeys((decod.get(m, {}) or {}).get(ch)) if decod else {}
        if ch == "image" and (0 not in acc or L not in acc) and args.phase4_image_root:
            model_dir = Path(args.phase4_image_root) / m
            for level in LEVELS:
                matches = sorted(model_dir.glob(f"level_{level}_*.json"))
                if len(matches) != 1:
                    continue
                with matches[0].open(encoding="utf-8") as handle:
                    phase4 = json.load(handle)
                if phase4.get("accuracy") is not None:
                    acc[level] = float(phase4["accuracy"])
        if 0 not in acc or L not in acc or acc[0] < 1e-6:
            return None, None
        headroom = acc[0] >= args.min_headroom
        return ((acc[0] - acc[L]) / acc[0]), headroom

    def survival_loss(m, ch, L):
        s = _intkeys((surv or {}).get(ch)) if surv else {}
        if 0 not in s or L not in s or (s[0] or 0) < 1e-6:
            return None, None
        return ((s[0] - s[L]) / s[0]), True   # survival L0==1.0, always has headroom

    providers = []
    if decod is not None:
        providers.append(("task_acc (single-modality VLM accuracy)", taskacc_loss))
    if surv is not None:
        providers.append(("survival (OCR/char survival)", survival_loss))

    for label, loss_fn in providers:
        pts = []
        for (m, ch, L), y in shifts.items():
            if L == 0:
                continue
            x, hr = loss_fn(m, ch, L)
            if x is None or not hr:
                continue
            pts.append((x, y, 1 if ch == "text" else 0))
        print(f"--- legibility axis: {label}  ({len(pts)} points) ---")
        # per-channel mean slope (descriptive)
        for mod, name in [(0, "image"), (1, "text")]:
            cp = [(x, y) for x, y, mm in pts if mm == mod]
            if len(cp) >= 2:
                import statistics as _st
                sl = _st.covariance([p[0] for p in cp], [p[1] for p in cp]) / \
                     (_st.variance([p[0] for p in cp]) or 1e-9)
                print(f"    {name:5s} slope ~ {sl:+.3f}  (n={len(cp)})")
        fit = ols_interaction(pts)
        if fit is None:
            print("    (not enough points / numpy+scipy needed for the interaction test)\n")
            continue
        print("    interaction model  arb_shift ~ b0 + b1*leg + b2*mod + b3*(mod*leg):")
        for nm, b, se, p in zip(fit["names"], fit["beta"], fit["se"], fit["p"]):
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"      {nm:16s} = {b:+.3f}  (se {se:.3f}, p={p:.2g}) {star}")
        b3, p3 = fit["beta"][3], fit["p"][3]
        verdict = ("residual TEXT-PRIMACY beyond legibility" if (p3 < 0.05 and b3 > 0)
                   else "no statistically resolved residual asymmetry" if p3 >= 0.05
                   else "residual IMAGE-primacy (unexpected)")
        print(f"    -> b3 (text slope - image slope) = {b3:+.3f}, p={p3:.2g}: {verdict}\n")

    print("Robustness: if b3 agrees in sign/significance across BOTH axes, the conclusion is")
    print("not an artifact of how legibility was operationalized.")


if __name__ == "__main__":
    main()
