#!/usr/bin/env python3
"""Single-modality decodability under degradation — the CONTROL axis for the
legibility asymmetry (Phase 6 image arm vs Phase 7 text arm).

The mismatch experiments show arbitration barely moves when the IMAGE degrades but
collapses when the TEXT degrades. A reviewer will object that the two corruption
ladders may not remove *equivalent* information. Rather than hand-calibrate the
corruptions to match (expensive, itself contestable), we MEASURE how much each channel
actually loses at each level, with NO conflict:

    image channel:  degrade the image at level L, ask image-only  -> accuracy_image(L)
    text  channel:  degrade the text  at level L, ask text-only   -> accuracy_text(L)

Accuracy = fraction solved correctly from that channel alone. This is the standard
psychophysics move: don't match the stimuli, measure the mediating variable
(legibility loss) and plot the arbitration shift against it. scripts/analyze_legibility_control.py
then joins this with the arbitration curves.

Image degradation uses src.noise (blur/noise ladder, seed=42+i, matching the image arm);
text degradation uses src.text_noise (char-corruption ladder, seed=i). Both are applied on
top of the canonical HF renders / questions so L0 == the clean baseline.

Usage:
    # one model, both channels, gsm8k (300-problem subset is plenty for an accuracy axis)
    python scripts/measure_legibility_decodability.py --models Qwen2.5-VL-7B-Instruct \
        --benchmark gsm8k --num-problems 300

    # one channel only; completed model/level cells are resumed safely
    python scripts/measure_legibility_decodability.py --models Qwen2.5-VL-7B-Instruct \
        --benchmark gsm8k --num-problems 200 --channels text

    # metadata-only sanity check (no model load, no HF download)
    python scripts/measure_legibility_decodability.py --dry-run --num-problems 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for run_legibility import

from src.evaluation import answers_match, compute_accuracy  # noqa: E402
from src.noise import NOISE_LEVELS, apply_noise_level        # noqa: E402
from src.text_noise import TEXT_NOISE_LEVELS, degrade_text    # noqa: E402
from src.benchmarks import load_benchmark                     # noqa: E402
from run_legibility import MODEL_REGISTRY, mismatch_prompt    # noqa: E402

IMAGE_SEED_BASE = 42
# Levels shared by both ladders (the monotonic legibility rungs).
DEFAULT_LEVELS = [0, 2, 4, 5]


def _atomic_write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def measure_model(model_key, questions, references, images, levels, channels, n,
                  out_dir, dry_run):
    """Return {'image': {L: acc}, 'text': {L: acc}, 'n': n} for one model."""
    mc = MODEL_REGISTRY[model_key]
    result_path = os.path.join(out_dir, f"{model_key}.json")
    result = {"model": model_key, "n": n, "image": {}, "text": {}}
    if os.path.exists(result_path):
        with open(result_path) as f:
            previous = json.load(f)
        if previous.get("model") != model_key or previous.get("n") != n:
            raise ValueError(
                f"{result_path} belongs to model={previous.get('model')}, "
                f"n={previous.get('n')}; refusing an incompatible resume"
            )
        result["image"].update(previous.get("image", {}))
        result["text"].update(previous.get("text", {}))

    if dry_run:
        # No model: just confirm the degradations assemble for each level.
        for L in levels:
            if "image" in channels and L in NOISE_LEVELS:
                result["image"][str(L)] = None
            if "text" in channels and L in TEXT_NOISE_LEVELS:
                _ = degrade_text(questions[0], L, seed=0)
                result["text"][str(L)] = None
        return result

    from src.models import VLMModel
    vlm = VLMModel(model_name=mc["name"], model_type=mc["type"],
                   max_new_tokens=256, torch_dtype="bfloat16")
    vlm.load()
    try:
        # ── image channel: image-only accuracy at each image level ──
        for L in [x for x in levels if "image" in channels and x in NOISE_LEVELS]:
            if str(L) in result["image"]:
                print(f"  {model_key} image L{L}: already complete, skipping")
                continue
            correct = []
            for i in tqdm(range(n), desc=f"{model_key} img L{L}"):
                try:
                    base = images[i]
                    if base is None:
                        raise ValueError("missing canonical image")
                    if base.mode != "RGB":
                        base = base.convert("RGB")
                    img = apply_noise_level(base, L, text=questions[i],
                                            seed=IMAGE_SEED_BASE + i)
                    pred = vlm.generate_with_image(img)  # image-only (default solve prompt)
                except Exception as e:
                    pred = f"ERROR: {e}"
                correct.append(answers_match(pred, references[i]))
            result["image"][str(L)] = compute_accuracy(correct)
            _atomic_write_json(result_path, result)

        # ── text channel: text-only accuracy on degraded text at each text level ──
        for L in [x for x in levels if "text" in channels and x in TEXT_NOISE_LEVELS]:
            if str(L) in result["text"]:
                print(f"  {model_key} text L{L}: already complete, skipping")
                continue
            correct = []
            for i in tqdm(range(n), desc=f"{model_key} txt L{L}"):
                try:
                    q = degrade_text(questions[i], L, seed=i)
                    pred = vlm.generate_text_only(q)  # text-only, corrupted text
                except Exception as e:
                    pred = f"ERROR: {e}"
                correct.append(answers_match(pred, references[i]))
            result["text"][str(L)] = compute_accuracy(correct)
            _atomic_write_json(result_path, result)
    finally:
        vlm.unload()
    return result


def main():
    ap = argparse.ArgumentParser(description="Single-modality decodability under degradation")
    ap.add_argument("--models", nargs="+", required=False, default=None,
                    help="model keys from the registry (default: all open + frontier)")
    ap.add_argument("--benchmark", default="gsm8k")
    ap.add_argument("--num-problems", type=int, default=300,
                    help="problems for the accuracy estimate (300 gives ~+-5%% CI)")
    ap.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    ap.add_argument("--channels", nargs="+", choices=["image", "text"],
                    default=["image", "text"],
                    help="single-modality channels to measure (default: both)")
    ap.add_argument("--output-dir", default="results/phase_control/decodability")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble degradations only; no model load, no HF images")
    args = ap.parse_args()

    models = args.models or list(MODEL_REGISTRY)
    out_dir = os.path.join(args.output_dir, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)

    use_hf = not args.dry_run  # dry-run skips the HF image download
    items = load_benchmark(args.benchmark, args.num_problems, use_hf=use_hf)
    questions = [it.question for it in items]
    references = [it.reference_answer for it in items]
    images = [getattr(it, "image", None) for it in items]
    n = len(questions)
    print(f"Loaded {n} {args.benchmark} problems (levels {args.levels})")

    all_res = {}
    for mk in models:
        if mk not in MODEL_REGISTRY:
            print(f"  unknown model '{mk}' — skipping"); continue
        res = measure_model(mk, questions, references, images, args.levels,
                            args.channels, n, out_dir, args.dry_run)
        all_res[mk] = res
        img = {L: (f"{v:.3f}" if v is not None else "-") for L, v in res["image"].items()}
        txt = {L: (f"{v:.3f}" if v is not None else "-") for L, v in res["text"].items()}
        print(f"  {mk:30s} image_acc={img}  text_acc={txt}")

    if not args.dry_run:
        # Rebuild from all compatible per-model files. This makes independent
        # one-model Slurm jobs safe: the aggregate never intentionally drops a
        # model completed by an earlier job.
        aggregate = {}
        for path in sorted(Path(out_dir).glob("*.json")):
            if path.name == "decodability_all.json":
                continue
            with path.open() as f:
                item = json.load(f)
            if item.get("model") and item.get("n") == n:
                aggregate[item["model"]] = item
        _atomic_write_json(os.path.join(out_dir, "decodability_all.json"), aggregate)
        print(f"\nDecodability written -> {out_dir}/")
    else:
        print("\n[dry-run] degradations assemble OK; no accuracies computed.")


if __name__ == "__main__":
    main()
