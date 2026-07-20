#!/usr/bin/env python3
"""Character-survival legibility — a model-INDEPENDENT legibility axis for the control
analysis (Phase 6/7 asymmetry), complementary to the single-modality *task accuracy* from
measure_legibility_decodability.py.

Two operationalizations of "how much text information physically survives" at each level:
  text channel:  char survival of degrade_text(q, L) vs q  -- deterministic, FREE (no model).
  image channel: OCR (Tesseract) the degraded rendered image, char survival vs q.

    char_survival = 1 - CER,   CER = Levenshtein(ref, hyp) / len(ref)

This measures PERCEPTUAL degradation (information recoverable by a neutral reader), separate
from what a reasoning VLM can exploit (task accuracy). Agreement between the two axes is a
strong robustness result; divergence localizes failure to perception vs exploitation.

The image channel needs `pytesseract` + the tesseract binary; the text channel needs neither
(use --text-only to skip OCR). Model-independent, so this is ONE run per benchmark, on CPU.

Usage:
    python scripts/measure_legibility_survival.py --benchmark svamp --num-problems 300
    python scripts/measure_legibility_survival.py --benchmark svamp --num-problems 5 --text-only
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

from src.noise import NOISE_LEVELS, apply_noise_level      # noqa: E402
from src.text_noise import TEXT_NOISE_LEVELS, degrade_text  # noqa: E402
from src.benchmarks import load_benchmark                   # noqa: E402

IMAGE_SEED_BASE = 42
DEFAULT_LEVELS = [0, 2, 4, 5]
_WS = re.compile(r"\s+")


def _norm(s):
    """Whitespace-collapse + lowercase so we measure content survival, not layout/case noise."""
    return _WS.sub(" ", (s or "").strip().lower())


def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


def char_survival(ref, hyp):
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return None
    return max(0.0, 1.0 - levenshtein(r, h) / len(r))


def main():
    ap = argparse.ArgumentParser(description="Character-survival legibility (model-independent)")
    ap.add_argument("--benchmark", default="svamp")
    ap.add_argument("--num-problems", type=int, default=300)
    ap.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    ap.add_argument("--output-dir", default="results/phase_control/survival")
    ap.add_argument("--text-only", action="store_true",
                    help="skip the image/OCR channel (no Tesseract needed)")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)

    need_images = not args.text_only
    items = load_benchmark(args.benchmark, args.num_problems, use_hf=need_images)
    questions = [it.question for it in items]
    images = [getattr(it, "image", None) for it in items]
    n = len(questions)
    print(f"Loaded {n} {args.benchmark} problems")

    res = {"benchmark": args.benchmark, "n": n, "metric": "char_survival",
           "text": {}, "image": {}}

    # ── text channel: deterministic, free ──
    for L in [x for x in args.levels if x in TEXT_NOISE_LEVELS]:
        vals = [char_survival(questions[i], degrade_text(questions[i], L, seed=i))
                for i in range(n)]
        vals = [v for v in vals if v is not None]
        res["text"][L] = st.mean(vals) if vals else None
    print("text  survival:", {L: round(v, 3) for L, v in res["text"].items() if v is not None})

    # ── image channel: OCR the degraded renders (Tesseract) ──
    if need_images:
        if any(im is None for im in images):
            print("WARNING: some canonical images missing — image channel may be partial.")
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            raise SystemExit("image channel needs `pip install pytesseract` + the tesseract "
                             "binary. Re-run with --text-only to skip OCR.")
        import pytesseract
        for L in [x for x in args.levels if x in NOISE_LEVELS]:
            vals = []
            for i in tqdm(range(n), desc=f"OCR L{L}"):
                base = images[i]
                if base is None:
                    continue
                if base.mode != "RGB":
                    base = base.convert("RGB")
                img = apply_noise_level(base, L, text=questions[i], seed=IMAGE_SEED_BASE + i)
                try:
                    rec = pytesseract.image_to_string(img)
                except Exception as e:
                    rec = ""
                s = char_survival(questions[i], rec)
                if s is not None:
                    vals.append(s)
            res["image"][L] = st.mean(vals) if vals else None
        print("image survival:", {L: round(v, 3) for L, v in res["image"].items() if v is not None})

    out = os.path.join(out_dir, "survival.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
