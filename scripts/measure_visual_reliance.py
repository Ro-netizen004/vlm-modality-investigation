#!/usr/bin/env python3
"""Protocol-B visual reliance-calibration probe -- does a VLM's reliance on a GENUINELY
visual channel track that channel's legibility?

Directly addresses the "rendered text != real visual reasoning" critique. If the same
reliability (in)sensitivity we see for rendered text also appears when the image is a real
chart/diagram, the finding generalizes beyond OCR. No conflict construction needed.

On a vision-essential benchmark (AI2D / ChartQA), present image+question at each image
degradation level and measure, per level:
    accuracy(L)     -- falls if the image is essential AND being read
    confidence(L)   -- mean generated-token logprob; a reliability-aware model should lose
                       confidence commensurate with accuracy
    invariance(L)   -- fraction of answers UNCHANGED from L0 (high under heavy blur => the
                       model ignores the degraded image / rides a text/prior)

Reliability-INSENSITIVITY signature (matching the rendered-text finding): accuracy collapses
while confidence stays ~flat and/or answers stay invariant -- the model keeps confidently
answering a visual channel it can no longer read. Reliability-AWARE signature: confidence
and answer-change track the accuracy drop.

Reuses the exact Protocol-B prompt + scoring (src.benchmark_eval), degrading item.image with
src.noise.apply_noise_level (seed=42+i, matching the rest of the pipeline).

Usage:
    python scripts/measure_visual_reliance.py --models Qwen2.5-VL-7B-Instruct --benchmark ai2d --num-problems 300
"""
import argparse
import json
import os
import re
import sys
import statistics as st
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmarks import load_benchmark             # noqa: E402
from src.benchmark_eval import match_answer           # noqa: E402
from src.noise import NOISE_LEVELS, apply_noise_level  # noqa: E402
from src.evaluation import compute_accuracy            # noqa: E402
from run_legibility import MODEL_REGISTRY              # noqa: E402

IMAGE_SEED_BASE = 42
DEFAULT_LEVELS = [0, 2, 4, 5]
# genuinely visual (vision-essential) Protocol-B benchmarks
VISUAL_BENCHMARKS = ["ai2d", "chartqa", "scienceqa", "mathvista"]


def build_prompt(item):
    p = item.question
    p += ("\nAnswer with the letter of the correct option." if item.choices
          else "\nProvide the answer. End with '#### <answer>'.")
    return p


def extract_answer(pred, item):
    if item.choices:
        m = re.search(r"\b([A-E])\b", (pred or "").upper())
        return m.group(1) if m else None
    m = re.search(r"####\s*([\-\d\.,]+)", pred or "")
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d+\.?\d*", pred or "")
    return nums[-1] if nums else None


def mean_conf(logprobs):
    if not logprobs:
        return None
    lps = [e.get("logprob") for e in logprobs if isinstance(e, dict) and e.get("logprob") is not None]
    return st.mean(lps) if lps else None


def measure_model(model_key, items, levels, out_dir):
    mc = MODEL_REGISTRY[model_key]
    from src.models import VLMModel
    vlm = VLMModel(model_name=mc["name"], model_type=mc["type"],
                   max_new_tokens=256, torch_dtype="bfloat16")
    vlm.load()
    res = {"model": model_key, "n": len(items), "levels": {}}
    l0_ans = {}
    try:
        for L in levels:
            correct, confs, ans = [], [], {}
            for i, item in enumerate(tqdm(items, desc=f"{model_key} L{L}")):
                try:
                    img = apply_noise_level(item.image, L, text=None, seed=IMAGE_SEED_BASE + i)
                    pred = vlm.generate_with_image(img, text_prompt=build_prompt(item))
                except Exception as e:
                    pred = f"ERROR: {e}"
                correct.append(match_answer(pred, item))
                c = mean_conf(getattr(vlm, "last_logprobs", None))
                if c is not None:
                    confs.append(c)
                ans[i] = extract_answer(pred, item)
                if L == 0:
                    l0_ans[i] = ans[i]
            if L == 0:
                inv = 1.0
            else:
                inv = st.mean([1.0 if ans[i] == l0_ans.get(i) else 0.0 for i in range(len(items))])
            res["levels"][L] = {
                "accuracy": compute_accuracy(correct),
                "mean_confidence": (st.mean(confs) if confs else None),
                "answer_invariance_vs_L0": inv,
                "n": len(items),
            }
            with open(os.path.join(out_dir, f"{model_key}.json"), "w") as f:
                json.dump(res, f, indent=2)
    finally:
        vlm.unload()
    return res


def main():
    ap = argparse.ArgumentParser(description="Protocol-B visual reliance-calibration probe")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--benchmark", default="ai2d", choices=VISUAL_BENCHMARKS)
    ap.add_argument("--num-problems", type=int, default=300)
    ap.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    ap.add_argument("--output-dir", default="results/phase_control/visual_reliance")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)
    items = [it for it in load_benchmark(args.benchmark, args.num_problems, use_hf=True)
             if getattr(it, "image", None) is not None]
    if not items:
        raise SystemExit(f"{args.benchmark}: no items with images loaded.")
    print(f"Loaded {len(items)} {args.benchmark} visual items (levels {args.levels})")

    models = args.models or list(MODEL_REGISTRY)
    all_res = {}
    for mk in models:
        if mk not in MODEL_REGISTRY:
            print(f"  unknown model '{mk}' — skipping"); continue
        r = measure_model(mk, items, args.levels, out_dir)
        all_res[mk] = r
        # quick trajectory read: accuracy drop vs confidence drop (L0 -> last level)
        lv = r["levels"]; Ls = sorted(lv)
        if 0 in lv and lv[Ls[-1]]:
            a0, aL = lv[0]["accuracy"], lv[Ls[-1]]["accuracy"]
            c0, cL = lv[0]["mean_confidence"], lv[Ls[-1]]["mean_confidence"]
            print(f"  {mk}: acc {a0:.3f}->{aL:.3f} (drop {a0-aL:+.3f}) | "
                  f"conf {c0}->{cL} | invariance@L{Ls[-1]}={lv[Ls[-1]]['answer_invariance_vs_L0']:.2f}")

    with open(os.path.join(out_dir, "visual_reliance_all.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nWrote {out_dir}/visual_reliance_all.json")
    print("Read-out: accuracy collapses while confidence stays flat / invariance stays high"
          "\n         => reliability-INSENSITIVITY on genuinely visual content (finding generalizes).")


if __name__ == "__main__":
    main()
