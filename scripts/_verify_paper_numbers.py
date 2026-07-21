#!/usr/bin/env python3
"""One-off audit: compare paper/main.tex tables against results/ on disk."""
import csv
import json
import math
import os
import re
import statistics as st
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAPER = {
    "tab:accuracy_gsm8k": {
        "Qwen2-VL-2B-Instruct": {"text": 0.426, "img": 0.400, "delta": -0.026, "h": 0.052},
        "llava-v1.6-mistral-7b-hf": {"text": 0.434, "img": 0.291, "delta": -0.143, "h": 0.299},
        "Qwen2.5-VL-7B-Instruct": {"text": 0.845, "img": 0.847, "delta": 0.002, "h": 0.004},
        "Idefics3-8B-Llama3": {"text": 0.773, "img": 0.280, "delta": -0.493, "h": 1.033},
        "MiniCPM-V-2_6": {"text": 0.748, "img": 0.601, "delta": -0.147, "h": 0.316},
        "InternVL2-8B": {"text": 0.771, "img": 0.776, "delta": 0.005, "h": 0.013},
        "llava-onevision-qwen2-7b-ov-hf": {"text": 0.790, "img": 0.626, "delta": -0.164, "h": 0.363},
        "Phi-3.5-vision-instruct": {"text": 0.776, "img": 0.727, "delta": -0.049, "h": 0.112},
    },
    "tab:mismatch": {
        "Qwen2-VL-2B-Instruct": {"TextPref": 0.983, "TextPref*": 0.988, "Neither*": 0.002},
        "llava-v1.6-mistral-7b-hf": {"TextPref": 0.989, "TextPref*": 0.994, "Neither*": 0.000},
        "Qwen2.5-VL-7B-Instruct": {"TextPref": 0.993, "TextPref*": 0.994, "Neither*": 0.000},
        "Idefics3-8B-Llama3": {"TextPref": 0.994, "TextPref*": 0.994, "Neither*": 0.020},
        "MiniCPM-V-2_6": {"TextPref": 0.994, "TextPref*": 0.995, "Neither*": 0.000},
        "InternVL2-8B": {"TextPref": 0.962, "TextPref*": 0.962, "Neither*": 0.001},
        "llava-onevision-qwen2-7b-ov-hf": {"TextPref": 0.998, "TextPref*": 0.998, "Neither*": 0.004},
        "Phi-3.5-vision-instruct": {"TextPref": 0.987, "TextPref*": 0.869, "Neither*": 0.058},
    },
    "tab:cll_replication_image": {
        "Idefics3-8B-Llama3": {"gsm8k_d": 0.86, "gsm8k_p": 1.4e-41, "svamp_d": 1.95, "svamp_p": 1.7e-33},
        "Qwen2.5-VL-7B-Instruct": {"gsm8k_d": 0.49, "gsm8k_p": 1.3e-146, "svamp_d": 0.92, "svamp_p": 2.9e-39},
        "Qwen2-VL-2B-Instruct": {"gsm8k_d": 0.06, "gsm8k_p": 3.8e-28, "svamp_d": 0.32, "svamp_p": 2.0e-23},
        "llava-onevision-qwen2-7b-ov-hf": {"gsm8k_d": 0.00, "gsm8k_p": 2.4e-1, "svamp_d": 0.01, "svamp_p": 5.2e-2},
        "llava-v1.6-mistral-7b-hf": {"gsm8k_d": 0.00, "gsm8k_p": 2.7e-1, "svamp_d": 0.17, "svamp_p": 4.2e-16},
        "Phi-3.5-vision-instruct": {"gsm8k_d": -0.05, "gsm8k_p": 6.4e-6, "svamp_d": -0.11, "svamp_p": 6.7e-5},
    },
    "tab:mirror_cll": {
        "Idefics3-8B-Llama3": {"gsm8k_d": -5.38, "gsm8k_p": 1.5e-147, "svamp_d": -9.16, "svamp_p": 1.7e-47},
        "Phi-3.5-vision-instruct": {"gsm8k_d": -1.32, "gsm8k_p": 5.5e-137, "svamp_d": -3.75, "svamp_p": 2.9e-42},
        "Qwen2-VL-2B-Instruct": {"gsm8k_d": -0.95, "gsm8k_p": 1.8e-129, "svamp_d": -1.27, "svamp_p": 9.6e-36},
        "Qwen2.5-VL-7B-Instruct": {"gsm8k_d": -1.44, "gsm8k_p": 7.6e-177, "svamp_d": -4.07, "svamp_p": 3.8e-48},
        "llava-onevision-qwen2-7b-ov-hf": {"gsm8k_d": -1.58, "gsm8k_p": 1.9e-176, "svamp_d": -3.27, "svamp_p": 3.5e-45},
        "llava-v1.6-mistral-7b-hf": {"gsm8k_d": None, "svamp_d": -1.61, "svamp_p": 2.4e-40},
    },
}

PAPER_B = {
    "Qwen2-VL-2B-Instruct": {"mathvista": 0.146, "ai2d": 0.158, "chartqa": 0.514, "scienceqa": 0.157},
    "llava-v1.6-mistral-7b-hf": {"mathvista": 0.046, "ai2d": 0.099, "chartqa": 0.288, "scienceqa": 0.109},
    "Qwen2.5-VL-7B-Instruct": {"mathvista": 0.260, "ai2d": 0.214, "chartqa": 0.628, "scienceqa": 0.200},
    "Idefics3-8B-Llama3": {"mathvista": 0.159, "ai2d": 0.158, "chartqa": 0.464, "scienceqa": 0.218},
    "MiniCPM-V-2_6": {"mathvista": 0.208, "ai2d": 0.116, "chartqa": 0.540, "scienceqa": 0.116},
    "InternVL2-8B": {"mathvista": 0.176, "ai2d": 0.234, "chartqa": 0.406, "scienceqa": 0.298},
    "llava-onevision-qwen2-7b-ov-hf": {"mathvista": 0.136, "ai2d": 0.171, "chartqa": 0.539, "scienceqa": 0.240},
}

PAPER_NOISE = {
    "Qwen2-VL-2B-Instruct": {"text": 0.530, "L0": 0.445, "L5": 0.230},
    "llava-v1.6-mistral-7b-hf": {"text": 0.430, "L0": 0.305, "L5": 0.005},
    "Idefics3-8B-Llama3": {"text": 0.790, "L0": 0.295, "L5": 0.025},
}

PAPER_DIS = {
    "Qwen2-VL-2B-Instruct": {"agree": 0.766, "text_adv": 171, "img_adv": 137},
    "Idefics3-8B-Llama3": {"agree": 0.445, "text_adv": 691, "img_adv": 41},
    "InternVL2-8B": {"agree": 0.767, "text_adv": 150, "img_adv": 157},
}

CLL_RE = re.compile(r"^level_(\d+)_.*\.cll\.jsonl$")
issues = []
ok = []


def chk(label, got, exp, tol=0.006):
    if exp is None or got is None or (isinstance(got, float) and math.isnan(got)):
        return
    if abs(got - exp) > tol:
        issues.append(f"MISMATCH {label}: got {got:.4f} paper {exp:.4f}")
    else:
        ok.append(label)


def load_stats(model):
    p = ROOT / "results/phase1" / model / "statistics.json"
    return json.loads(p.read_text()) if p.exists() else None


def median_margins(root, model, bench):
    per = {}
    mdir = root / bench / model
    if not mdir.is_dir():
        return per
    for fn in os.listdir(mdir):
        m = CLL_RE.match(fn)
        if not m:
            continue
        level = int(m.group(1))
        vals = []
        with open(mdir / fn, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                mm = (row.get("margin") or {}).get("margin_mean")
                if mm is not None:
                    vals.append((int(row["i"]), mm))
        if vals:
            per[level] = [value for _, value in sorted(vals)]
    return per


def cll_stats(root, model, bench):
    per = median_margins(root, model, bench)
    if 0 not in per or 5 not in per:
        return None
    # median_margins sorts both levels by stable item id.
    n = min(len(per[0]), len(per[5]))
    diffs = [per[5][i] - per[0][i] for i in range(n)]
    _, pu = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    d = st.median(diffs)
    return d, pu, st.median(per[0]), st.median(per[5])


def main():
    from src.evaluation import cohens_h

    print("=" * 70)
    print("PHASE 1: GSM8K accuracy + mismatch")
    print("=" * 70)
    for model, exp in PAPER["tab:accuracy_gsm8k"].items():
        s = load_stats(model)
        if not s:
            issues.append(f"MISSING phase1 stats for {model}")
            continue
        chk(f"{model} text", s["acc_text"], exp["text"])
        chk(f"{model} img", s["acc_img"], exp["img"])
        chk(f"{model} delta", s["acc_img"] - s["acc_text"], exp["delta"])
        chk(f"{model} h", cohens_h(s["acc_text"], s["acc_img"]), exp["h"], 0.015)

    for model, exp in PAPER["tab:mismatch"].items():
        s = load_stats(model)
        if not s:
            continue
        chk(f"{model} TextPref", s["text_preference"], exp["TextPref"])
        r = s.get("rescore", {})
        chk(f"{model} TextPref*", r.get("text_preference_rescored"), exp["TextPref*"])
        neither_rate = r.get("neither_final", 0) / s["n"]
        chk(f"{model} Neither*", neither_rate, exp["Neither*"], 0.003)

    print("\n" + "=" * 70)
    print("PHASE 3: Protocol B image advantage")
    print("=" * 70)
    for model, benches in PAPER_B.items():
        for bench, exp in benches.items():
            p = ROOT / "results/phase3" / model / bench / "statistics.json"
            if not p.exists():
                issues.append(f"MISSING phase3 {model}/{bench}")
                continue
            s = json.loads(p.read_text())
            got = s.get("acc_diff", s.get("acc_multimodal", 0) - s.get("acc_text_only", 0))
            chk(f"ProtocolB {model} {bench}", got, exp)

    print("\n" + "=" * 70)
    print("PHASE 6/7: CLL margin tables")
    print("=" * 70)
    for model, exp in PAPER["tab:cll_replication_image"].items():
        for bench in ("gsm8k", "svamp"):
            got = cll_stats(ROOT / "results/phase6_legibility", model, bench)
            if got is None:
                issues.append(f"MISSING image CLL {model}/{bench}")
                continue
            d, pval, m0, m5 = got
            chk(f"image CLL {model} {bench} d", d, exp[f"{bench}_d"], 0.05)
            if exp[f"{bench}_p"] > 0 and pval > 0:
                if abs(math.log10(pval) - math.log10(exp[f"{bench}_p"])) > 1.5:
                    issues.append(
                        f"MISMATCH image CLL {model} {bench} p: got {pval:.2e} paper {exp[f'{bench}_p']:.2e}"
                    )
                else:
                    ok.append(f"image CLL {model} {bench} p")
            print(f"  image {model} {bench}: d={d:+.2f} med L0={m0:.2f} L5={m5:.2f} p={pval:.2e}")

    for model, exp in PAPER["tab:mirror_cll"].items():
        for bench in ("gsm8k", "svamp"):
            if exp[f"{bench}_d"] is None:
                print(f"  text CLL {model} {bench}: PENDING in paper")
                continue
            got = cll_stats(ROOT / "results/phase7_text_legibility", model, bench)
            if got is None:
                issues.append(f"MISSING text CLL {model}/{bench}")
                continue
            d, pval, m0, m5 = got
            chk(f"text CLL {model} {bench} d", d, exp[f"{bench}_d"], 0.15)
            if exp[f"{bench}_p"] > 0 and pval > 0:
                if abs(math.log10(pval) - math.log10(exp[f"{bench}_p"])) > 2:
                    issues.append(
                        f"MISMATCH text CLL {model} {bench} p: got {pval:.2e} paper {exp[f'{bench}_p']:.2e}"
                    )
                else:
                    ok.append(f"text CLL {model} {bench} p")
            print(f"  text {model} {bench}: d={d:+.2f} med L0={m0:.2f} L5={m5:.2f} p={pval:.2e}")

    print("\n" + "=" * 70)
    print("CLL sign agreement (paper: 0.82, n=15991)")
    print("=" * 70)
    pooled_ok = pooled_n = 0
    for bench in ("gsm8k", "svamp"):
        bdir = ROOT / "results/phase6_legibility" / bench
        if not bdir.is_dir():
            continue
        for model in os.listdir(bdir):
            mdir = bdir / model
            if not mdir.is_dir():
                continue
            for fn in os.listdir(mdir):
                if not fn.endswith(".cll.jsonl"):
                    continue
                with open(mdir / fn, encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        if row.get("follows") in ("text", "image") and row.get("margin"):
                            pred = "text" if row["margin"]["margin_mean"] > 0 else "image"
                            pooled_n += 1
                            pooled_ok += pred == row["follows"]
    if pooled_n:
        agr = pooled_ok / pooled_n
        print(f"  computed: {agr:.3f} n={pooled_n}")
        if abs(agr - 0.82) > 0.02 or abs(pooled_n - 15991) > 50:
            issues.append(f"MISMATCH sign agreement: got {agr:.3f} n={pooled_n}, paper 0.82 n=15991")
        else:
            ok.append("sign agreement")

    per = median_margins(ROOT / "results/phase6_legibility", "Qwen2.5-VL-7B-Instruct", "gsm8k")
    if 0 in per and 5 in per:
        m0, m5 = st.median(per[0]), st.median(per[5])
        print(f"\nQwen2.5-VL GSM8K margin L0={m0:.2f} L5={m5:.2f} (paper 0.78 -> 1.54)")
        if abs(m0 - 0.78) > 0.1 or abs(m5 - 1.54) > 0.15:
            issues.append(f"MISMATCH Qwen2.5 margin trajectory: got {m0:.2f}->{m5:.2f}")

    fig = ROOT / "paper/figures/margin_vs_legibility.png"
    print(f"\nFigure margin_vs_legibility.png exists: {fig.exists()}")

    print("\n" + "=" * 70)
    print("PHASE 4: noise appendix table (subset)")
    print("=" * 70)
    for model, exp in PAPER_NOISE.items():
        summ = ROOT / f"results/phase4/{model}/noise_summary.json"
        if not summ.exists():
            issues.append(f"MISSING noise summary {model}")
            continue
        s = json.loads(summ.read_text())
        tb = json.loads((ROOT / f"results/phase4/{model}/text_baseline.json").read_text())
        got_text = tb.get("accuracy")
        l0 = s["levels"]["0"]["accuracy"]
        l5 = s["levels"]["5"]["accuracy"]
        chk(f"noise {model} text", got_text, exp["text"], 0.01)
        chk(f"noise {model} L0", l0, exp["L0"], 0.01)
        chk(f"noise {model} L5", l5, exp["L5"], 0.01)

    print("\n" + "=" * 70)
    print("Disagreement table")
    print("=" * 70)
    for model, exp in PAPER_DIS.items():
        s = load_stats(model)
        if not s:
            issues.append(f"MISSING stats for disagree {model}")
            continue
        n = s["n"]
        text_adv = s["mcnemar_b"]
        img_adv = s["mcnemar_c"]
        agree = 1 - (text_adv + img_adv) / n
        chk(f"disagree {model} agree", agree, exp["agree"], 0.01)
        if text_adv != exp["text_adv"]:
            issues.append(f"MISMATCH disagree {model} text_adv: got {text_adv} paper {exp['text_adv']}")
        if img_adv != exp["img_adv"]:
            issues.append(f"MISMATCH disagree {model} img_adv: got {img_adv} paper {exp['img_adv']}")

    # Text arm behavioral claims
    print("\n" + "=" * 70)
    print("Mirror arm behavioral (paper prose)")
    print("=" * 70)
    p7 = json.loads((ROOT / "results/phase7_text_legibility/svamp/legibility_all_rescored.json").read_text())
    q = p7["Qwen2.5-VL-7B-Instruct"]["text_preference_by_level"]
    got = q["0"], q["5"]
    print(f"  Qwen2.5 SVAMP text pref L0->L5 rescored: {got[0]:.3f}->{got[1]:.3f} (paper 0.99->0.29)")
    if abs(got[0] - 0.99) > 0.02 or abs(got[1] - 0.29) > 0.05:
        issues.append(f"MISMATCH Qwen2.5 SVAMP text pref rescored: {got[0]:.3f}->{got[1]:.3f}")

    # beta3 interaction
    print("\n" + "=" * 70)
    print("Control regression beta3 (paper: +6.4 p=0.042 gsm8k, +8.5 p=0.11 svamp)")
    print("=" * 70)
    try:
        import subprocess

        for bm in ("gsm8k", "svamp"):
            r = subprocess.run(
                ["python", "scripts/analyze_legibility_control.py", "--benchmark", bm],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            print(r.stdout[-800:] if len(r.stdout) > 800 else r.stdout)
            if r.returncode != 0:
                issues.append(f"analyze_legibility_control failed for {bm}: {r.stderr[:200]}")
    except Exception as e:
        issues.append(f"Could not run control analysis: {e}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(ok)} checks passed, {len(issues)} issues")
    print("=" * 70)
    for item in issues:
        print("  !", item)
    if not issues:
        print("  All checked numbers match within tolerance.")


if __name__ == "__main__":
    main()
