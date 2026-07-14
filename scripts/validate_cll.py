"""Validate conditional_loglik / arbitration_margin on a few mismatch trials.

Run on GAIVI (GPU). The key correctness check: the CLL margin sign (text vs image)
should mostly AGREE with what the model actually generates under the same conflict.
If agreement is ~chance (0.5), the scoring is wrong (tokenization boundary, scaffold
mismatch, etc.) — do NOT scale it. If it's high (>~0.75), the CLL is measuring the
right thing and the graded margin is trustworthy on the ceiling models.

Usage:
    python scripts/validate_cll.py --model Qwen2-VL-2B-Instruct --benchmark gsm8k --n 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import VLMModel
from src.benchmarks import load_benchmark
from src.evaluation import score_mismatch_follows

# _generate-family models only (CLL needs forward-pass access; not InternVL2/MiniCPM).
REGISTRY = {
    "Qwen2-VL-2B-Instruct":           ("Qwen/Qwen2-VL-2B-Instruct",          "qwen"),
    "Qwen2.5-VL-7B-Instruct":         ("Qwen/Qwen2.5-VL-7B-Instruct",        "qwen"),
    "Phi-3.5-vision-instruct":        ("microsoft/Phi-3.5-vision-instruct",  "phi"),
    "Idefics3-8B-Llama3":             ("HuggingFaceM4/Idefics3-8B-Llama3",   "idefics"),
    "llava-v1.6-mistral-7b-hf":       ("llava-hf/llava-v1.6-mistral-7b-hf",  "llava"),
    "llava-onevision-qwen2-7b-ov-hf": ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava_onevision"),
}

MISMATCH_PROMPT = ("Solve the following math problem step by step. "
                   "End with '#### <answer>'.\n\nProblem: {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2-VL-2B-Instruct", choices=list(REGISTRY))
    ap.add_argument("--benchmark", default="gsm8k")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    name, mtype = REGISTRY[args.model]
    vlm = VLMModel(model_name=name, model_type=mtype,
                   max_new_tokens=256, torch_dtype="bfloat16")
    vlm.load()

    items = load_benchmark(args.benchmark, args.n + 1, use_hf=True)
    n = len(items)
    imgs = [it.image for it in items]
    refs = [it.reference_answer for it in items]

    agree = total = 0
    print(f"\n{'i':>3} {'follows':>8} {'margin':>8} {'CLL_txt':>8} {'CLL_img':>8}  txt/img ans")
    for i in range(min(args.n, n)):
        txt_idx = (i + 1) % n           # image = problem i, text = problem txt_idx
        img = imgs[i]
        if img is None:
            continue
        m = vlm.arbitration_margin(img, text_answer=refs[txt_idx], image_answer=refs[i],
                                   text_question=items[txt_idx].question)
        gen = vlm.generate_with_image(img, text_prompt=MISMATCH_PROMPT.format(q=items[txt_idx].question))
        follows = score_mismatch_follows(gen, refs[i], refs[txt_idx])  # (pred, img_ref, txt_ref)
        if m is None:
            print(f"{i:>3}  margin=None (scoring returned None)")
            continue
        pred_side = "text" if m["margin_mean"] > 0 else "image"
        if follows in ("text", "image"):
            total += 1
            agree += (pred_side == follows)
        print(f"{i:>3} {follows:>8} {m['margin_mean']:>+8.3f} {m['cll_text_mean']:>8.3f} "
              f"{m['cll_image_mean']:>8.3f}  {refs[txt_idx]}/{refs[i]}")

    if total:
        print(f"\nmargin-sign vs generation agreement: {agree}/{total} = {agree/total:.2f}")
        print("  >0.75 → CLL is trustworthy; ~0.5 → scoring is broken, do not scale.")
    vlm.unload()


if __name__ == "__main__":
    main()
