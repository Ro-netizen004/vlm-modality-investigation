#!/usr/bin/env python3
"""
Legibility experiment — modality preference under image degradation.

Runs the MISMATCH condition (image of problem i + text of problem i+1) at each
image-corruption level, measuring how text preference changes as the image
becomes less legible. A flat curve indicates a fixed text bias insensitive to
input quality; a sloped curve indicates rational, reliability-aware arbitration.

Reuses the noisy images already rendered by the Phase 4 noise ablation
(results/phase4/images/level_<L>_<name>/q###.png), so no re-rendering is needed
if those exist.

Designed to be fanned out across the cluster: run one (model, level) per SLURM
job (see scripts/gaivi_run_legibility_parallel.sh). Each job is self-contained
and checkpoints per problem, so a job killed at the wall-time limit resumes
where it left off instead of restarting the level.

Usage:
    # single cell (one model, one level) — the fan-out unit
    python scripts/run_legibility.py --models Idefics3-8B-Llama3 --noise-levels 5

    # a few models / levels serially (local / debugging)
    python scripts/run_legibility.py --models Qwen2.5-VL-7B-Instruct --noise-levels 0 2 4 5

    # merge per-level JSONs into per-model + combined summaries (after the grid finishes)
    python scripts/run_legibility.py --merge \
        --models Idefics3-8B-Llama3 Qwen2.5-VL-7B-Instruct
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import VLMModel
from src.evaluation import score_mismatch_follows
from src.noise import NOISE_LEVELS, render_noisy_images, apply_noise_to_images
from src.benchmarks import load_benchmark

# Legibility only makes sense on Protocol A (text rendered as an image). We also
# restrict to NUMERIC-answer benchmarks, because score_mismatch_follows attributes
# a trial to image/text by numeric match. AQuA-RAT is multiple-choice (letter
# answers) so it's excluded — its conflict attribution would be unreliable.
TEXT_MATH_BENCHMARKS = ["gsm8k", "svamp", "math"]

# Same registry/types as the noise runner — must match configs/default.yaml.
MODEL_REGISTRY = {
    "Qwen2-VL-2B-Instruct":          {"name": "Qwen/Qwen2-VL-2B-Instruct",          "type": "qwen"},
    "llava-v1.6-mistral-7b-hf":      {"name": "llava-hf/llava-v1.6-mistral-7b-hf",  "type": "llava"},
    "Qwen2.5-VL-7B-Instruct":        {"name": "Qwen/Qwen2.5-VL-7B-Instruct",        "type": "qwen"},
    "Idefics3-8B-Llama3":            {"name": "HuggingFaceM4/Idefics3-8B-Llama3",   "type": "idefics"},
    "MiniCPM-V-2_6":                 {"name": "openbmb/MiniCPM-V-2_6",              "type": "minicpm"},
    "InternVL2-8B":                  {"name": "OpenGVLab/InternVL2-8B",             "type": "internvl"},
    "llava-onevision-qwen2-7b-ov-hf":{"name": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "type": "llava_onevision"},
    "Phi-3.5-vision-instruct":       {"name": "microsoft/Phi-3.5-vision-instruct",  "type": "phi"},
}
DEFAULT_MODELS = ["Idefics3-8B-Llama3", "Qwen2.5-VL-7B-Instruct"]  # vulnerable + resilient anchors
ALL_MODELS = list(MODEL_REGISTRY.keys())  # full 8-model set for the headline run


def mismatch_prompt(text_question):
    return ("Solve the following math problem step by step. "
            "End with '#### <answer>'.\n\n"
            f"Problem: {text_question}")


def _atomic_write_json(path, obj):
    """Write JSON via temp-file + os.replace so concurrent jobs never see a
    half-written file (the per-model summary is touched by every level job)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _summarize(follows, n):
    """Build the level result dict from a list of per-problem 'follows' labels."""
    counts = Counter(follows)
    n_img = counts.get("image", 0)
    n_txt = counts.get("text", 0)
    decidable = n_img + n_txt
    return {
        "counts": dict(counts),
        "decidable": decidable,
        "text_preference": (n_txt / decidable) if decidable else None,
        "image_preference": (n_img / decidable) if decidable else None,
        "neither_rate": counts.get("neither", 0) / n if n else None,
    }


def run_level(vlm, level, questions, references, image_dir, n, out_dir):
    """Run the mismatch condition for all problems at one noise level.

    Idempotent + resumable:
      * if the final level JSON already exists, it's loaded and returned (skip);
      * otherwise per-problem results are appended to a .partial.jsonl as they
        complete, and a restarted job skips problems already recorded there.
    """
    config = NOISE_LEVELS[level]
    name = config["name"]
    final_path = os.path.join(out_dir, f"level_{level}_{name}.json")
    if os.path.exists(final_path):
        with open(final_path) as f:
            return json.load(f)

    partial_path = os.path.join(out_dir, f"level_{level}_{name}.partial.jsonl")
    done = {}
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[int(rec["i"])] = rec["follows"]
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a torn last line from a hard kill
        print(f"  L{level} {name}: resuming, {len(done)}/{n} already done")

    level_img_dir = os.path.join(image_dir, f"level_{level}_{name}")
    follows = [None] * n

    with open(partial_path, "a") as pf:
        for i in tqdm(range(n), desc=f"L{level}"):
            if i in done:
                follows[i] = done[i]
                continue
            txt_idx = (i + 1) % n
            img_path = os.path.join(level_img_dir, f"q{i:03d}.png")
            try:
                img = Image.open(img_path).convert("RGB")
                pred = vlm.generate_with_image(
                    img, text_prompt=mismatch_prompt(questions[txt_idx]))
            except Exception as e:
                pred = f"ERROR: {e}"
            label = score_mismatch_follows(pred, references[i], references[txt_idx])
            follows[i] = label
            pf.write(json.dumps({"i": i, "follows": label}) + "\n")
            pf.flush()

    res = {"level": level, "name": name, **_summarize(follows, n)}
    _atomic_write_json(final_path, res)
    try:
        os.remove(partial_path)  # checkpoint no longer needed once level is final
    except OSError:
        pass
    return res


def rebuild_model_summary(out_dir, model_key, n):
    """Reconstruct the per-model summary from whatever level_*.json files exist.

    Safe to call from any level job — it reflects all completed levels so far,
    and the write is atomic, so concurrent level jobs can't corrupt it.
    """
    levels_found = {}
    for fn in os.listdir(out_dir):
        m = re.match(r"level_(\d+)_.*\.json$", fn)
        if not m:
            continue
        try:
            with open(os.path.join(out_dir, fn)) as f:
                levels_found[int(m.group(1))] = json.load(f)
        except json.JSONDecodeError:
            continue

    ordered = sorted(levels_found)
    summary = {
        "model": model_key,
        "n": n,
        "levels_complete": ordered,
        "text_preference_by_level": {L: levels_found[L]["text_preference"] for L in ordered},
        "neither_rate_by_level": {L: levels_found[L]["neither_rate"] for L in ordered},
    }
    _atomic_write_json(os.path.join(out_dir, "legibility_summary.json"), summary)

    tps = [levels_found[L]["text_preference"] for L in ordered
           if levels_found[L]["text_preference"] is not None]
    if tps:
        print(f"  [{model_key}] text preference across {len(ordered)} levels: "
              f"{min(tps):.3f}--{max(tps):.3f} (spread {max(tps) - min(tps):.3f})")
    return summary


def run_model(model_key, questions, references, image_dir, n, levels, out_root):
    mc = MODEL_REGISTRY[model_key]
    out_dir = os.path.join(out_root, model_key)
    os.makedirs(out_dir, exist_ok=True)

    remaining = [
        L for L in levels
        if not os.path.exists(os.path.join(out_dir, f"level_{L}_{NOISE_LEVELS[L]['name']}.json"))
    ]
    if not remaining:
        print(f"\n{'='*60}\n  {model_key}: all requested levels done — merging only\n{'='*60}")
        return rebuild_model_summary(out_dir, model_key, n)

    print(f"\n{'='*60}\n  {model_key}  (levels to run: {remaining})\n{'='*60}")
    vlm = VLMModel(model_name=mc["name"], model_type=mc["type"],
                   max_new_tokens=256, torch_dtype="bfloat16")
    vlm.load()

    for level in levels:  # run_level self-skips any already-final level
        res = run_level(vlm, level, questions, references, image_dir, n, out_dir)
        tp = res["text_preference"]
        if tp is not None:
            print(f"  L{level} {res['name']:18s}: text_pref={tp:.3f}  "
                  f"decidable={res['decidable']}  neither={res['neither_rate']*100:.1f}%")
        else:
            print(f"  L{level} {res['name']}: no decidable trials")
        rebuild_model_summary(out_dir, model_key, n)  # refresh after each level

    vlm.unload()
    return rebuild_model_summary(out_dir, model_key, n)


def main():
    parser = argparse.ArgumentParser(description="Legibility experiment (mismatch x noise)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmark", default="gsm8k", choices=TEXT_MATH_BENCHMARKS,
                        help="Which Protocol-A (rendered-text) benchmark to run.")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--noise-levels", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--noise-image-dir", default=None,
                        help="Where the noisy images live (reused if present). "
                             "Defaults to a per-benchmark dir under the output dir. "
                             "Noise is applied on top of the canonical HF images so "
                             "Level 0 matches the images used in the main experiments.")
    parser.add_argument("--output-dir", default="results/phase6_legibility")
    parser.add_argument("--merge", action="store_true",
                        help="Skip inference; just rebuild per-model + combined summaries "
                             "from existing level_*.json files.")
    parser.add_argument("--render-only", action="store_true",
                        help="Only render the noisy images for this benchmark, then exit. "
                             "Run this once before fanning out compute jobs so they don't "
                             "race to render the same files.")
    args = parser.parse_args()

    torch.manual_seed(42)

    # Namespace outputs by benchmark. gsm8k stays at the root for backward
    # compatibility (existing results + launcher/plotter paths); other benchmarks
    # get their own subdir so results never collide.
    out_root = (args.output_dir if args.benchmark == "gsm8k"
                else os.path.join(args.output_dir, args.benchmark))
    os.makedirs(out_root, exist_ok=True)

    # ── Merge-only mode: collate the fanned-out per-level results ──
    if args.merge:
        all_summaries = {}
        for model_key in args.models:
            out_dir = os.path.join(out_root, model_key)
            if os.path.isdir(out_dir):
                all_summaries[model_key] = rebuild_model_summary(
                    out_dir, model_key, args.num_problems)
            else:
                print(f"  {model_key}: no results dir — skipping")
        _atomic_write_json(os.path.join(out_root, "legibility_all.json"), all_summaries)
        print(f"\nMerge complete → {out_root}/legibility_all.json")
        return

    # Load the CANONICAL HF renders (use_hf=True) so noise is applied on top of the
    # exact images used in the main experiments — Level 0 is then pixel-identical to
    # the Phase 1/3 mismatch baseline, per the CLAUDE.md "use canonical PNGs" rule.
    items = load_benchmark(args.benchmark, args.num_problems, use_hf=True)
    questions = [it.question for it in items]
    references = [it.reference_answer for it in items]
    images = [it.image for it in items]
    n = len(questions)
    print(f"Loaded {n} {args.benchmark} problems (canonical HF images)")
    if args.benchmark == "math":
        print("  Note: MATH answers are often non-numeric (fractions/expressions); "
              "expect a smaller decidable set than GSM8K/SVAMP.")

    image_dir = args.noise_image_dir or os.path.join(out_root, "noise_images")
    have_images = all(
        os.path.isdir(os.path.join(image_dir, f"level_{L}_{NOISE_LEVELS[L]['name']}"))
        for L in args.noise_levels
    )
    if not have_images:
        os.makedirs(image_dir, exist_ok=True)
        if all(img is not None for img in images):
            print(f"Applying noise to canonical HF images → {image_dir} ...")
            apply_noise_to_images(images, image_dir,
                                  noise_levels=args.noise_levels, texts=questions)
        else:
            missing = sum(img is None for img in images)
            print(f"WARNING: {missing} canonical images missing — falling back to "
                  f"fresh text render (Level 0 may not match the main experiments).")
            render_noisy_images(questions, image_dir, noise_levels=args.noise_levels)
    else:
        print(f"Reusing existing noisy images from {image_dir}")

    if args.render_only:
        print(f"Render-only: images ready in {image_dir}. Exiting.")
        return

    all_summaries = {}
    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model '{model_key}' — skipping")
            continue
        all_summaries[model_key] = run_model(
            model_key, questions, references, image_dir, n,
            args.noise_levels, out_root)

    _atomic_write_json(os.path.join(out_root, "legibility_all.json"), all_summaries)
    print(f"\nLegibility experiment complete ({args.benchmark}).")


if __name__ == "__main__":
    main()
