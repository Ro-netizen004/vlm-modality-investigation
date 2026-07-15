"""Analyze the CLL arbitration margins: scale validation, ceiling-cracker curve,
2x2 reasoned-vs-gut dissociation, and coverage. Reads results/cll_gaivi/*.cll.jsonl."""
import json, os, glob, statistics as st
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "cll_gaivi")
LEVELS = [(0, "clean"), (2, "blur_light"), (4, "blur_noise"), (5, "heavy_degradation")]
LEVEL_FILE = {0: "level_0_clean", 2: "level_2_blur_light",
              4: "level_4_blur_noise", 5: "level_5_heavy_degradation"}

def load(model, level):
    p = os.path.join(ROOT, model, f"{LEVEL_FILE[level]}.cll.jsonl")
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("margin"):
                rows.append(r)
    return rows

models = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))

# ── 1. SCALE VALIDATION: margin sign vs generation follows ─────────────────────
print("="*72)
print("1. SCALE VALIDATION  — margin-sign vs generation 'follows' (decidable only)")
print("="*72)
pooled_ok = pooled_n = 0
for m in models:
    ok = tot = 0
    for L, _ in LEVELS:
        for r in load(m, L):
            if r["follows"] in ("text", "image"):
                pred = "text" if r["margin"]["margin_mean"] > 0 else "image"
                tot += 1; ok += (pred == r["follows"])
    pooled_ok += ok; pooled_n += tot
    print(f"  {m:32s} {ok:>5}/{tot:<5} = {ok/tot:.3f}")
print(f"  {'POOLED':32s} {pooled_ok:>5}/{pooled_n:<5} = {pooled_ok/pooled_n:.3f}")
# Wilson 95% CI on pooled
import math
p = pooled_ok/pooled_n; z = 1.96; nn = pooled_n
den = 1+z*z/nn
lo = (p+z*z/(2*nn) - z*math.sqrt(p*(1-p)/nn + z*z/(4*nn*nn)))/den
hi = (p+z*z/(2*nn) + z*math.sqrt(p*(1-p)/nn + z*z/(4*nn*nn)))/den
print(f"  pooled 95% CI: [{lo:.3f}, {hi:.3f}]   (n={pooled_n})")

# ── 2. CEILING-CRACKER: binary text-pref vs graded CLL, per level ──────────────
print("\n" + "="*72)
print("2. CEILING-CRACKER  — binary text-pref vs CLL margin, by legibility level")
print("   binary_tp = P(follow text | decidable);  cll_txt% = P(margin>0);")
print("   med_margin = median(CLL_text - CLL_image).  Does the graded signal move")
print("   where the binary is pinned?")
print("="*72)
for m in models:
    print(f"\n  {m}")
    print(f"    {'lvl':>4}{'binary_tp':>11}{'cll_txt%':>10}{'med_margin':>12}{'mean_margin':>12}")
    for L, _ in LEVELS:
        rows = load(m, L)
        dec = [r for r in rows if r["follows"] in ("text", "image")]
        btp = sum(r["follows"] == "text" for r in dec)/len(dec) if dec else float("nan")
        margins = [r["margin"]["margin_mean"] for r in rows]
        cll_txt = sum(x > 0 for x in margins)/len(margins)
        print(f"    {L:>4}{btp:>11.3f}{cll_txt:>10.3f}{st.median(margins):>12.3f}{st.mean(margins):>12.3f}")

# ── 3. DISSOCIATION 2x2 (reasoning x CLL sign), pooled per level ───────────────
print("\n" + "="*72)
print("3. DISSOCIATION 2x2  — rows: reasoning (image/text);  cols: CLL (image/text)")
print("   off-diagonal = reasoning diverges from immediate (gut) preference")
print("="*72)
for L, name in LEVELS:
    cell = Counter()
    for m in models:
        for r in load(m, L):
            rs = r["reasoning"]
            if rs not in ("text_reasoning", "image_reasoning"):
                continue
            rs = "text" if rs == "text_reasoning" else "image"
            cs = "text" if r["margin"]["margin_mean"] > 0 else "image"
            cell[(rs, cs)] += 1
    tot = sum(cell.values())
    off = cell[("image", "text")] + cell[("text", "image")]
    print(f"\n  Level {L} ({name}):  n={tot}")
    print(f"                      CLL:image   CLL:text")
    print(f"    reason:image     {cell[('image','image')]:>9}  {cell[('image','text')]:>9}   (stable-vis / reason-overrides-vis)")
    print(f"    reason:text      {cell[('text','image')]:>9}  {cell[('text','text')]:>9}   (reason-overrides-txt / stable-txt)")
    print(f"    divergence (off-diagonal): {off}/{tot} = {off/tot:.3f}" if tot else "    n/a")

# ── 4. COVERAGE: does CLL rescue the 'neither' trials? ─────────────────────────
print("\n" + "="*72)
print("4. COVERAGE  — of generation-'neither' trials, how many get a decisive CLL")
print("   margin (|margin_mean| > 1.0)?")
print("="*72)
for m in models:
    nei = dec = 0
    for L, _ in LEVELS:
        for r in load(m, L):
            if r["follows"] == "neither":
                nei += 1
                dec += (abs(r["margin"]["margin_mean"]) > 1.0)
    print(f"  {m:32s} {dec:>5}/{nei:<5} neither rescued = {dec/nei:.3f}" if nei else f"  {m}: no neither")

# ── 5. TRAJECTORY SIGNIFICANCE: does the margin shift with legibility? ──────────
from scipy import stats
print("\n" + "="*72)
print("5. TRAJECTORY SIGNIFICANCE  — MWU: L0 vs L5 margin;  Spearman(margin, level)")
print("="*72)
for m in models:
    per_level = {L: [r["margin"]["margin_mean"] for r in load(m, L)] for L, _ in LEVELS}
    l0, l5 = per_level[0], per_level[5]
    u, pu = stats.mannwhitneyu(l5, l0, alternative="two-sided")
    dmed = st.median(l5) - st.median(l0)
    xs, ys = [], []
    for L, _ in LEVELS:
        xs += [L] * len(per_level[L]); ys += per_level[L]
    rho, prho = stats.spearmanr(xs, ys)
    rises = dmed > 0 and pu < 0.05 and rho > 0 and prho < 0.05
    flag = "RISES (ceiling-cracked)" if rises else ("flat" if pu >= 0.05 else "shifts(non-monotone)")
    print(f"  {m:32s} d_med={dmed:+.3f}  MWU p={pu:.1e}  Spearman rho={rho:+.3f} (p={prho:.1e})  [{flag}]")
