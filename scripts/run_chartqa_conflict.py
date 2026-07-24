#!/usr/bin/env python3
"""Natural-visual conflict control on ChartQA.

Each item has one clean shared question and two conflicting evidence sources:
the attached chart supports the ChartQA gold answer A, while a textual report
explicitly supports a deterministic counterfactual B. The image and report are
degraded in separate arms under a neutral, source-label-counterbalanced prompt.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_legibility import CLL_TYPES, MODEL_REGISTRY
from src.benchmarks import load_benchmark
from src.models import VLMModel
from src.noise import NOISE_LEVELS, apply_noise_level
from src.text_noise import TEXT_NOISE_LEVELS, degrade_text


DESIGN_VERSION = "chartqa-same-question-conflict-v4"
LEVELS = (0, 2, 4, 5)
DEFAULT_DATASET_REPO = "vlm-modality-research/chartqa-evidence-conflict-v1"
DEFAULT_DATASET_REVISION = "3ead711196b4bf75ae6c23be8148bb8417047c4e"
FROZEN_MANIFEST_SHA256 = (
    "388bce0572487024f5ac12261621cbab8931ec3032d8bf0c65a258c134d20842"
)
YES_SYNONYMS = {"yes", "true", "correct", "affirmative"}
NO_SYNONYMS = {"no", "false", "incorrect", "negative"}
CURRENCY = {"$": "usd", "€": "eur", "£": "gbp", "¥": "jpy"}
UNIT_ALIASES = {
    "%": "percent", "percent": "percent", "percentage": "percent",
    "dollar": "usd", "dollars": "usd", "usd": "usd",
    "euro": "eur", "euros": "eur", "eur": "eur",
    "pound": "gbp", "pounds": "gbp", "gbp": "gbp",
    "yen": "jpy", "jpy": "jpy",
}


def atomic_json(path: Path, value) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temp, path)


def manifest_digest(rows) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_simple_answer(answer):
    text = str(answer).strip()
    lower = text.lower().rstrip(".")
    if lower in {"yes", "no"}:
        return "boolean", lower
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(%)?\s*", text.replace(",", ""))
    if match:
        return "numeric", (float(match.group(1)), bool(match.group(2)), match.group(1))
    return None, None


def counterfactual(answer):
    kind, value = parse_simple_answer(answer)
    if kind == "boolean":
        return ("no" if value == "yes" else "yes"), kind
    if kind != "numeric":
        return None, None
    number, percent, raw = value
    delta = max(1.0, abs(number) * 0.20)
    altered = number + delta
    decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
    if decimals == 0:
        altered = float(round(altered))
        if altered == number:
            altered += 1.0
        rendered = str(int(altered))
    else:
        altered = round(altered, decimals)
        if altered == number:
            altered = round(altered + 10 ** (-decimals), decimals)
        rendered = f"{altered:.{decimals}f}"
    if percent:
        rendered += "%"
    return rendered, kind


def build_manifest(items, n, seed):
    eligible = []
    for item in items:
        alternate, kind = counterfactual(item.reference_answer)
        if alternate is None or getattr(item, "image", True) is None:
            continue
        eligible.append({
            "dataset_index": int(item.id),
            "question": item.question,
            "image_answer": str(item.reference_answer).strip(),
            "text_answer": alternate,
            "answer_type": kind,
            "report_type": "assertion",
            "counterfactual_strategy": "boolean_flip" if kind == "boolean" else "draft_perturbation",
            "text_report": f'A separate textual report states: "The answer is {alternate}."',
        })
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:n]
    if len(selected) < n:
        raise RuntimeError(f"Only {len(selected)} eligible ChartQA items; requested {n}")
    for index, row in enumerate(selected):
        row["conflict_id"] = index
        row["image_label"], row["text_label"] = (
            ("A", "B") if index % 2 == 0 else ("B", "A")
        )
    return selected


def load_chartqa_metadata(arrow_path=None):
    """Load exact HF row indices without decoding 2,500 images for manifest preparation."""
    dataset = Dataset.from_file(str(arrow_path)) if arrow_path else load_dataset(
        "lmms-lab/ChartQA", split="test"
    )
    if "image" in dataset.column_names:
        dataset = dataset.remove_columns("image")
    return [SimpleNamespace(
        id=index,
        question=row.get("question", row.get("query", "")),
        reference_answer=str(row.get("answer", row.get("label", ""))),
    ) for index, row in enumerate(dataset)]


def load_published_conflict_dataset(repo, revision, expected_n):
    """Load the frozen derivative dataset and reconstruct its manifest exactly."""
    dataset = load_dataset(repo, split="test", revision=revision)
    if len(dataset) != expected_n:
        raise RuntimeError(
            f"Published dataset has {len(dataset)} rows; expected {expected_n}"
        )
    embedded_hashes = set(dataset["manifest_sha256"])
    if embedded_hashes != {FROZEN_MANIFEST_SHA256}:
        raise RuntimeError(
            f"Unexpected embedded manifest hashes: {sorted(embedded_hashes)}"
        )
    manifest, items_by_id = [], {}
    for row in dataset:
        dataset_index = int(row["chartqa_test_index"])
        manifest.append({
            "conflict_id": int(row["conflict_id"]),
            "pool_conflict_id": int(row["pool_conflict_id"]),
            "dataset_index": dataset_index,
            "question": row["question"],
            "image_answer": row["chart_answer"],
            "text_answer": row["report_answer"],
            "answer_type": row["answer_type"],
            "image_label": row["chart_source_label"],
            "text_label": row["report_source_label"],
            "report_type": "evidence",
            "counterfactual_strategy": row["counterfactual_strategy"],
            "unit_class": row["unit_class"],
            "text_report": row["text_report"],
            "source_table": row["source_table"],
            "evidence_validation": {
                "entailed": True,
                "counterfactual_valid": True,
                "reviewer": row["reviewer"],
                "notes": row["review_notes"],
            },
        })
        # Several QA rows may share one chart; the image is identical for that index.
        items_by_id[dataset_index] = SimpleNamespace(
            id=dataset_index, image=row["image"]
        )
    digest = manifest_digest(manifest)
    if digest != FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            f"Reconstructed manifest hash {digest} does not match frozen "
            f"{FROZEN_MANIFEST_SHA256}"
        )
    return manifest, items_by_id


def report_text(row):
    report = row.get("text_report")
    if not report:
        raise ValueError(f"Manifest row {row.get('conflict_id')} has no text_report")
    return report


def validate_manifest(manifest, report_type):
    errors = []
    for index, row in enumerate(manifest):
        if row.get("report_type") != report_type:
            errors.append(f"row {index}: report_type={row.get('report_type')!r}")
        if not str(row.get("text_report", "")).strip():
            errors.append(f"row {index}: empty text_report")
        if report_type == "evidence":
            if not row.get("evidence_validation", {}).get("entailed"):
                errors.append(f"row {index}: evidence report is not marked entailed")
            if not row.get("evidence_validation", {}).get("counterfactual_valid"):
                errors.append(f"row {index}: counterfactual is not marked valid/coherent")
            if row.get("counterfactual_strategy") not in {
                "chart_value", "nearby_category_value", "rank_swap",
                "arithmetic_alternative", "unit_preserving_perturbation", "boolean_flip",
            }:
                errors.append(f"row {index}: missing/invalid counterfactual strategy")
            if not row.get("source_table"):
                errors.append(f"row {index}: missing source_table provenance")
    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"Manifest failed {report_type} validation:\n{preview}")


def neutral_prompt(row, report, chain_of_thought):
    image_line = f"Source {row['image_label']} is the attached chart."
    text_line = f"Source {row['text_label']} is the textual report below:\n{report}"
    lines = [image_line, text_line] if row["image_label"] == "A" else [text_line, image_line]
    # Generation is deliberately answer-only.  Asking for a rationale caused some
    # models to exhaust their output budget before emitting the marked answer and
    # makes strict source attribution needlessly dependent on trace parsing.
    ending = (
        "Respond with exactly one line in the form '#### <answer>'. "
        "After ####, give only the answer value; do not explain your reasoning."
    )
    return (
        f"Question: {row['question']}\n\n"
        "You are given two conflicting evidence sources. Neither source is privileged.\n"
        + "\n".join(lines) + "\n\n" + ending
    )


def extract_final_answer(prediction):
    """Return only the explicitly marked final answer, or a terse answer-only output."""
    text = str(prediction).strip()
    marked = re.search(r"####\s*([^\r\n]+)", text)
    if marked:
        return marked.group(1).strip()
    # Accept explicit answer markers used by models that ignore the requested
    # delimiter.  These patterns remain anchored to the whole response or its
    # final clause; they never mine an arbitrary number from a reasoning trace.
    answer_line = re.fullmatch(r"(?i)(?:final\s+)?answer\s*:\s*(.+)", text)
    if answer_line:
        return answer_line.group(1).strip()
    terminal_answer = re.search(
        r"(?i)(?:therefore,?\s*)?(?:the\s+)?(?:final\s+)?answer\s+is\s+"
        r"([^\r\n.]+)\.?\s*$",
        text,
    )
    if terminal_answer:
        return terminal_answer.group(1).strip()
    # Do not mine reasoning traces for numbers. Accept only a single short answer line.
    if "\n" not in text and len(text.split()) <= 4:
        return text
    return None


def normalize_answer(answer, unit_hint=""):
    """Strict canonicalization; no distance, substring, or fuzzy matching."""
    if answer is None:
        return None
    text = str(answer).strip().lower().replace("−", "-").replace("–", "-")
    text = text.strip(" \t\r\n\"'`.")
    if text in YES_SYNONYMS:
        return ("boolean", "yes")
    if text in NO_SYNONYMS:
        return ("boolean", "no")

    pattern = re.fullmatch(
        r"(?P<currency>[$€£¥])?\s*(?P<number>[+-]?(?:[\d,]+(?:\.\d+)?|\d+\s*/\s*\d+))"
        r"\s*(?P<unit>%|[a-z]+)?",
        text,
    )
    if not pattern:
        return None
    raw_number = pattern.group("number").replace(",", "").replace(" ", "")
    try:
        if "/" in raw_number:
            value = Decimal(Fraction(raw_number).numerator) / Decimal(Fraction(raw_number).denominator)
        else:
            value = Decimal(raw_number)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    currency = CURRENCY.get(pattern.group("currency"))
    raw_unit = pattern.group("unit")
    unit = UNIT_ALIASES.get(raw_unit, raw_unit) if raw_unit else None
    hinted = UNIT_ALIASES.get(str(unit_hint).strip().lower(), str(unit_hint).strip().lower())
    if currency and unit and currency != unit:
        return None
    explicit_unit = currency or unit
    if explicit_unit and hinted and explicit_unit != hinted:
        return None
    return ("numeric", value.normalize(), explicit_unit or hinted or "unitless")


def classify(prediction, row):
    final = extract_final_answer(prediction)
    predicted = normalize_answer(final, row.get("unit_class", ""))
    image = normalize_answer(row["image_answer"], row.get("unit_class", ""))
    text = normalize_answer(row["text_answer"], row.get("unit_class", ""))
    if predicted is None or image is None or text is None:
        return "invalid", final, predicted
    if image == text and predicted == image:
        return "ambiguous", final, predicted
    if predicted == image:
        return "image", final, predicted
    if predicted == text:
        return "text", final, predicted
    return "neither", final, predicted


def load_jsonl(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                rows[int(row["i"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return rows


def run_level(vlm, items_by_id, manifest, model_dir, arm, level, mode):
    name = TEXT_NOISE_LEVELS[level]["name"] if arm == "text" else NOISE_LEVELS[level]["name"]
    path = model_dir / f"level_{level}_{name}.{mode}.jsonl"
    done = load_jsonl(path)
    with path.open("a", encoding="utf-8") as output:
        for i, row in enumerate(tqdm(manifest, desc=f"{mode} {arm} L{level}")):
            if i in done:
                continue
            item = items_by_id[row["dataset_index"]]
            image = item.image.convert("RGB")
            report = report_text(row)
            if arm == "image":
                image = apply_noise_level(image, level, text=None, seed=42 + i)
            elif level != 0:
                report = degrade_text(report, level, seed=i)
            prompt = neutral_prompt(row, report, chain_of_thought=(mode == "generation"))
            try:
                if mode == "generation":
                    prediction = vlm.generate_with_image(image, text_prompt=prompt)
                    follows, extracted, normalized = classify(prediction, row)
                    result = {"prediction": prediction, "extracted_final": extracted,
                              "normalized_final": repr(normalized), "follows": follows}
                else:
                    margin = vlm.candidate_margin(
                        image, row["text_answer"], row["image_answer"], prompt
                    )
                    result = {"margin": margin}
            except Exception as error:
                result = {"error": repr(error)}
                if mode == "generation":
                    result.update(prediction="", follows="invalid")
                else:
                    result["margin"] = None
            record = {"i": i, "dataset_index": row["dataset_index"], "level": level,
                      "arm": arm, "design_version": DESIGN_VERSION, **result}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


def summarize(model_dir, arm, mode):
    levels = {}
    for level in LEVELS:
        name = TEXT_NOISE_LEVELS[level]["name"] if arm == "text" else NOISE_LEVELS[level]["name"]
        path = model_dir / f"level_{level}_{name}.{mode}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            continue
        if mode == "generation":
            counts = Counter(row.get("follows", "invalid") for row in rows.values())
            decidable = counts["image"] + counts["text"]
            levels[level] = {"n": len(rows), "counts": dict(counts),
                             "text_preference": counts["text"] / decidable if decidable else None}
        else:
            valid = [row for row in rows.values() if (row.get("margin") or {}).get("margin_mean") is not None]
            levels[level] = {"n": len(rows), "valid_margins": len(valid)}
    atomic_json(model_dir / f"summary_{mode}.json",
                {"arm": arm, "mode": mode, "levels": levels})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--arm", choices=("image", "text"))
    parser.add_argument("--mode", choices=("generation", "cll"))
    parser.add_argument("--levels", nargs="+", type=int, default=list(LEVELS))
    parser.add_argument("--num-problems", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phase_control/chartqa_conflict"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report-type", choices=("evidence", "assertion"),
                        default="evidence")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--metadata-arrow", type=Path, default=None,
                        help="Optional cached ChartQA Arrow file for fast offline preparation")
    parser.add_argument("--dataset-repo", default=None,
                        help="Load the frozen conflict rows/images from this HF dataset")
    parser.add_argument("--dataset-revision", default=None,
                        help="Pinned HF dataset commit; required with --dataset-repo")
    args = parser.parse_args()

    if any(level not in LEVELS for level in args.levels):
        raise SystemExit(f"Levels must be drawn from {LEVELS}")
    condition_root = args.output_dir / args.report_type
    published_items = None
    if args.dataset_repo:
        if not args.dataset_revision:
            raise RuntimeError("--dataset-revision is required with --dataset-repo")
        manifest, published_items = load_published_conflict_dataset(
            args.dataset_repo, args.dataset_revision, args.num_problems
        )
        manifest_path = (
            f"hf://datasets/{args.dataset_repo}@{args.dataset_revision}"
        )
    else:
        manifest_path = args.manifest or condition_root / "manifest.json"
    if not args.dataset_repo and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if len(manifest) != args.num_problems:
            raise RuntimeError(f"Manifest has {len(manifest)} items; requested {args.num_problems}")
    elif not args.dataset_repo and args.report_type == "assertion":
        items = (load_chartqa_metadata(args.metadata_arrow) if args.prepare_only
                 else load_benchmark("chartqa", None))
        manifest = build_manifest(items, args.num_problems, args.seed)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    elif not args.dataset_repo:
        raise RuntimeError(
            "The evidence-bearing main condition requires a prebuilt --manifest with "
            "text_report, source_table, and evidence_validation. Use the official "
            "ChartQA tables to prepare it; --report-type assertion is only the ablation."
        )
    validate_manifest(manifest, args.report_type)
    print(f"Manifest: {manifest_path} n={len(manifest)} sha256={manifest_digest(manifest)}")
    if args.prepare_only:
        return
    if not args.models or args.arm is None or args.mode is None:
        parser.error("--models, --arm, and --mode are required unless --prepare-only is used")
    if published_items is not None:
        items_by_id = published_items
    else:
        items = load_benchmark("chartqa", None)
        items_by_id = {int(item.id): item for item in items}

    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model {model_key}; skipping")
            continue
        if args.mode == "cll" and MODEL_REGISTRY[model_key]["type"] not in CLL_TYPES:
            print(f"{model_key}: no candidate CLL support; skipping")
            continue
        model_dir = condition_root / args.arm / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        config = {"design_version": DESIGN_VERSION, "model": model_key, "arm": args.arm,
                  "mode": args.mode, "report_type": args.report_type, "n": len(manifest),
                  "manifest_sha256": manifest_digest(manifest),
                  "dataset_repo": args.dataset_repo,
                  "dataset_revision": args.dataset_revision}
        config_path = model_dir / f"config_{args.mode}.json"
        if config_path.exists() and json.load(config_path.open()) != config:
            raise RuntimeError(f"Incompatible existing configuration: {config_path}")
        atomic_json(config_path, config)
        spec = MODEL_REGISTRY[model_key]
        vlm = VLMModel(model_name=spec["name"], model_type=spec["type"],
                       max_new_tokens=128, torch_dtype="bfloat16")
        vlm.load()
        try:
            for level in args.levels:
                run_level(vlm, items_by_id, manifest, model_dir, args.arm, level, args.mode)
        finally:
            vlm.unload()
        summarize(model_dir, args.arm, args.mode)


if __name__ == "__main__":
    main()
