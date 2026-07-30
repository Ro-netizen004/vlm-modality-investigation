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
import tempfile
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from PIL import Image

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


def load_paired_representation_dataset(
    repo, revision, expected_n, visual_representation
):
    """Load the pinned chart/table release and select one visual representation."""
    dataset = load_dataset(repo, split="test", revision=revision)
    if len(dataset) != expected_n:
        raise RuntimeError(
            f"Paired representation dataset has {len(dataset)} rows; "
            f"expected {expected_n}"
        )
    required = {
        "conflict_id", "pool_conflict_id", "chartqa_test_index",
        "chart_image", "table_image", "question", "chart_answer",
        "report_answer", "text_report", "answer_type", "unit_class",
        "counterfactual_strategy", "chart_source_label",
        "report_source_label", "source_table", "official_table_data",
        "table_image_sha256", "provenance_scope",
        "provenance_audit_sha256", "original_manifest_sha256",
    }
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise RuntimeError(f"Paired dataset is missing columns: {missing}")
    original_hashes = set(dataset["original_manifest_sha256"])
    if original_hashes != {FROZEN_MANIFEST_SHA256}:
        raise RuntimeError(
            f"Unexpected original manifest hashes: {sorted(original_hashes)}"
        )
    audit_hashes = set(dataset["provenance_audit_sha256"])
    if len(audit_hashes) != 1:
        raise RuntimeError(
            f"Expected one provenance audit hash; found {sorted(audit_hashes)}"
        )

    image_column = (
        "chart_image" if visual_representation == "chart" else "table_image"
    )
    manifest, items_by_id = [], {}
    seen_conflict_ids = set()
    for row in dataset:
        conflict_id = int(row["conflict_id"])
        if conflict_id in seen_conflict_ids:
            raise RuntimeError(f"Duplicate conflict_id in paired dataset: {conflict_id}")
        seen_conflict_ids.add(conflict_id)
        dataset_index = int(row["chartqa_test_index"])
        manifest.append({
            "conflict_id": conflict_id,
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
                "reviewer": "frozen curated dataset",
                "notes": (
                    f"provenance_scope={row['provenance_scope']}; "
                    f"audit_sha256={row['provenance_audit_sha256']}"
                ),
            },
        })
        items_by_id[dataset_index] = SimpleNamespace(
            id=dataset_index, image=row[image_column].convert("RGB")
        )
    return manifest, items_by_id


def load_table_ablation_manifest(path: Path, expected_n: int):
    """Load rendered table images while preserving the frozen conflict metadata."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != expected_n:
        raise RuntimeError(
            f"Table manifest has {len(rows)} items; requested {expected_n}"
        )

    manifest, items_by_id = [], {}
    for index, row in enumerate(rows):
        required = {
            "conflict_id", "pool_conflict_id", "chartqa_test_index", "question",
            "chart_answer", "report_answer", "answer_type", "unit_class",
            "counterfactual_strategy", "text_report", "chart_source_label",
            "report_source_label", "source_table", "table_image_file",
            "table_image_sha256",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise RuntimeError(
                f"Table manifest row {index} is missing fields: {missing}"
            )
        image_path = (path.parent / row["table_image_file"]).resolve()
        if not image_path.is_file():
            raise RuntimeError(f"Missing table image: {image_path}")
        image_bytes = image_path.read_bytes()
        actual_hash = hashlib.sha256(image_bytes).hexdigest()
        if actual_hash != row["table_image_sha256"]:
            raise RuntimeError(
                f"Table image checksum mismatch for conflict {row['conflict_id']}: "
                f"{actual_hash} != {row['table_image_sha256']}"
            )

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
                "reviewer": "frozen curated dataset",
                "notes": "Metadata inherited from the audited ChartQA-Conflict item.",
            },
        })
        items_by_id[dataset_index] = SimpleNamespace(
            id=dataset_index, image=Image.open(image_path).convert("RGB").copy()
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


def neutral_prompt(row, report, chain_of_thought, visual_representation="chart"):
    visual_name = "chart" if visual_representation == "chart" else "table image"
    image_line = f"Source {row['image_label']} is the attached {visual_name}."
    text_line = f"Source {row['text_label']} is the textual report below:\n{report}"
    lines = [image_line, text_line] if row["image_label"] == "A" else [text_line, image_line]
    # Generation is deliberately answer-only.  Asking for a rationale caused some
    # models to exhaust their output budget before emitting the marked answer and
    # makes strict source attribution needlessly dependent on trace parsing.
    ending = (
        "Select exactly one source-supported answer, even though the sources disagree. "
        "Do not abstain, hedge, combine the answers, or list alternatives. "
        "Respond with exactly one line in the form '#### <answer>'. "
        "After ####, copy only the bare answer value (for example, 3.89 or 12.8%); "
        "do not include words, source names, or an explanation."
    )
    return (
        f"Question: {row['question']}\n\n"
        "You are given two conflicting evidence sources. Neither source is privileged.\n"
        + "\n".join(lines) + "\n\n" + ending
    )


def unimodal_prompt(row, arm, report=None, visual_representation="chart"):
    ending = (
        "Respond with exactly one line in the form '#### <answer>'. "
        "After ####, give only the answer value; do not explain your reasoning."
    )
    if arm == "image":
        visual_name = "chart" if visual_representation == "chart" else "table image"
        evidence = f"Use the attached {visual_name} as the only evidence source."
    else:
        evidence = f"Use the following report as the only evidence source:\n{report}"
    return f"Question: {row['question']}\n\n{evidence}\n\n{ending}"


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


def reuse_image_l0_for_text_arm(
    condition_root, model_key, model_dir, expected, mode
):
    """Reuse the identical clean endpoint instead of making a second API call.

    The L0 chart and report are identical in both arms. Reusing the image-arm
    generations removes API nondeterminism from the arm contrast.
    """
    source = (
        condition_root / "image" / model_key /
        f"level_0_clean.{mode}.jsonl"
    )
    target = model_dir / f"level_0_clean.{mode}.jsonl"
    rows = load_jsonl(source)
    if len(rows) != expected:
        raise RuntimeError(
            f"Cannot reuse L0: expected {expected} rows in {source}; found {len(rows)}"
        )
    if target.exists():
        existing = load_jsonl(target)
        if len(existing) == expected:
            return
        raise RuntimeError(
            f"Refusing to overwrite incomplete text-arm L0 file: {target}"
        )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=model_dir, suffix=".jsonl.tmp"
    ) as handle:
        temp_path = Path(handle.name)
        for i in sorted(rows):
            record = dict(rows[i])
            record["arm"] = "text"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp_path, target)
    print(f"Reused image-arm L0 {mode} rows -> {target}")


def run_level(
    vlm, items_by_id, manifest, model_dir, arm, level, mode,
    visual_representation="chart",
):
    name = TEXT_NOISE_LEVELS[level]["name"] if arm == "text" else NOISE_LEVELS[level]["name"]
    path = model_dir / f"level_{level}_{name}.{mode}.jsonl"
    done = load_jsonl(path)
    with path.open("a", encoding="utf-8") as output:
        for row in tqdm(manifest, desc=f"{mode} {arm} L{level}"):
            # Use the frozen conflict ID rather than the row's position. This
            # preserves identical corruption seeds when an audited subset is run.
            item_index = int(row["conflict_id"])
            if item_index in done:
                continue
            item = items_by_id[row["dataset_index"]]
            image = item.image.convert("RGB")
            report = report_text(row)
            if arm == "image":
                image = apply_noise_level(
                    image, level, text=None, seed=42 + item_index
                )
            elif level != 0:
                report = degrade_text(report, level, seed=item_index)
            try:
                if mode == "generation":
                    prompt = neutral_prompt(
                        row, report, chain_of_thought=True,
                        visual_representation=visual_representation,
                    )
                    prediction = vlm.generate_with_image(image, text_prompt=prompt)
                    follows, extracted, normalized = classify(prediction, row)
                    result = {"prediction": prediction, "extracted_final": extracted,
                              "normalized_final": repr(normalized), "follows": follows,
                              "api_response_meta": getattr(
                                  vlm, "last_response_meta", None
                              )}
                elif mode == "cll":
                    prompt = neutral_prompt(
                        row, report, chain_of_thought=False,
                        visual_representation=visual_representation,
                    )
                    margin = vlm.candidate_margin(
                        image, row["text_answer"], row["image_answer"], prompt
                    )
                    result = {"margin": margin}
                else:
                    prompt = unimodal_prompt(
                        row, arm, report=report if arm == "text" else None,
                        visual_representation=visual_representation,
                    )
                    prediction = (
                        vlm.generate_with_image(image, text_prompt=prompt)
                        if arm == "image"
                        else vlm.generate_text_only(prompt)
                    )
                    extracted = extract_final_answer(prediction)
                    normalized = normalize_answer(
                        extracted, row.get("unit_class", "")
                    )
                    target_answer = (
                        row["image_answer"] if arm == "image" else row["text_answer"]
                    )
                    target = normalize_answer(
                        target_answer, row.get("unit_class", "")
                    )
                    result = {
                        "prediction": prediction,
                        "extracted_final": extracted,
                        "normalized_final": repr(normalized),
                        "target_answer": target_answer,
                        "correct": normalized is not None and normalized == target,
                    }
            except Exception as error:
                result = {"error": repr(error)}
                if mode in {"generation", "decodability"}:
                    result.update(prediction="", follows="invalid")
                else:
                    result["margin"] = None
            record = {"i": item_index, "dataset_index": row["dataset_index"],
                      "level": level,
                      "arm": arm, "design_version": DESIGN_VERSION,
                      "visual_representation": visual_representation, **result}
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
        elif mode == "cll":
            valid = [row for row in rows.values() if (row.get("margin") or {}).get("margin_mean") is not None]
            levels[level] = {"n": len(rows), "valid_margins": len(valid)}
        else:
            correct = sum(bool(row.get("correct")) for row in rows.values())
            errors = sum("error" in row for row in rows.values())
            levels[level] = {
                "n": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "errors": errors,
            }
    atomic_json(model_dir / f"summary_{mode}.json",
                {"arm": arm, "mode": mode, "levels": levels})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--arm", choices=("image", "text"))
    parser.add_argument("--mode", choices=("generation", "cll", "decodability"))
    parser.add_argument("--levels", nargs="+", type=int, default=list(LEVELS))
    parser.add_argument("--num-problems", type=int, default=300)
    parser.add_argument(
        "--api-output-tokens", type=int, default=1024,
        help="Completion-token budget for frontier API models (default: 1024).",
    )
    parser.add_argument(
        "--openai-reasoning-effort", default="none",
        help="Reasoning effort for OpenAI API models; use 'default' to omit it.",
    )
    parser.add_argument(
        "--gemini-thinking-level", default="minimal",
        choices=("minimal", "low", "medium", "high", "default"),
        help="Thinking level for Gemini API models; 'default' omits the setting.",
    )
    parser.add_argument(
        "--reuse-image-l0", action="store_true",
        help=(
            "For text-arm generation or CLL, copy the identical image-arm L0 "
            "rows instead of recomputing them."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phase_control/chartqa_conflict"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--visual-representation",
        choices=("chart", "plain_table"),
        default="chart",
        help="Visual evidence format. plain_table requires --table-manifest.",
    )
    parser.add_argument(
        "--table-manifest",
        type=Path,
        default=None,
        help="JSONL manifest produced by prepare_chartqa_table_ablation.py.",
    )
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
    if (
        args.visual_representation == "plain_table"
        and not args.table_manifest
        and not args.dataset_repo
    ):
        parser.error(
            "--visual-representation plain_table requires --table-manifest "
            "or --dataset-repo"
        )
    if args.visual_representation == "chart" and args.table_manifest:
        parser.error("--table-manifest requires --visual-representation plain_table")
    if args.table_manifest and (args.dataset_repo or args.manifest):
        parser.error(
            "--table-manifest cannot be combined with --dataset-repo or --manifest"
        )
    condition_root = args.output_dir / args.report_type
    published_items = None
    table_items = None
    if args.table_manifest:
        manifest, table_items = load_table_ablation_manifest(
            args.table_manifest, args.num_problems
        )
        manifest_path = args.table_manifest
    elif args.dataset_repo:
        if not args.dataset_revision:
            raise RuntimeError("--dataset-revision is required with --dataset-repo")
        if args.visual_representation == "plain_table":
            manifest, published_items = load_paired_representation_dataset(
                args.dataset_repo,
                args.dataset_revision,
                args.num_problems,
                args.visual_representation,
            )
        else:
            # A paired release can also supply the original chart condition.
            probe = load_dataset(
                args.dataset_repo,
                split="test",
                revision=args.dataset_revision,
            )
            if {"chart_image", "table_image"}.issubset(probe.column_names):
                manifest, published_items = load_paired_representation_dataset(
                    args.dataset_repo,
                    args.dataset_revision,
                    args.num_problems,
                    args.visual_representation,
                )
            else:
                manifest, published_items = load_published_conflict_dataset(
                    args.dataset_repo, args.dataset_revision, args.num_problems
                )
        manifest_path = (
            f"hf://datasets/{args.dataset_repo}@{args.dataset_revision}"
        )
    else:
        manifest_path = args.manifest or condition_root / "manifest.json"
    if not args.dataset_repo and not args.table_manifest and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if len(manifest) != args.num_problems:
            raise RuntimeError(f"Manifest has {len(manifest)} items; requested {args.num_problems}")
    elif (
        not args.dataset_repo
        and not args.table_manifest
        and args.report_type == "assertion"
    ):
        items = (load_chartqa_metadata(args.metadata_arrow) if args.prepare_only
                 else load_benchmark("chartqa", None))
        manifest = build_manifest(items, args.num_problems, args.seed)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    elif not args.dataset_repo and not args.table_manifest:
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
    if table_items is not None:
        items_by_id = table_items
    elif published_items is not None:
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
                  "api_output_tokens": args.api_output_tokens,
                  "openai_reasoning_effort": args.openai_reasoning_effort,
                  "gemini_thinking_level": args.gemini_thinking_level,
                  "reuse_image_l0": args.reuse_image_l0,
                  "visual_representation": args.visual_representation,
                  "table_manifest_name": (
                      args.table_manifest.name
                      if args.table_manifest else None
                  ),
                  "table_manifest_sha256": (
                      hashlib.sha256(args.table_manifest.read_bytes()).hexdigest()
                      if args.table_manifest else None
                  ),
                  "manifest_sha256": manifest_digest(manifest),
                  "dataset_repo": args.dataset_repo,
                  "dataset_revision": args.dataset_revision}
        config_path = model_dir / f"config_{args.mode}.json"
        if config_path.exists() and json.load(config_path.open()) != config:
            raise RuntimeError(f"Incompatible existing configuration: {config_path}")
        atomic_json(config_path, config)
        spec = MODEL_REGISTRY[model_key]
        is_api = spec["type"] in {"openai", "gemini"}
        if args.reuse_image_l0:
            if args.arm != "text" or args.mode not in {"generation", "cll"}:
                parser.error(
                    "--reuse-image-l0 requires --arm text and "
                    "--mode generation or cll"
                )
            reuse_image_l0_for_text_arm(
                condition_root, model_key, model_dir, len(manifest), args.mode
            )
        vlm_kwargs = {
            "model_name": spec["name"],
            "model_type": spec["type"],
            "max_new_tokens": args.api_output_tokens if is_api else 128,
            "torch_dtype": "bfloat16",
        }
        if spec["type"] == "openai":
            vlm_kwargs["openai_reasoning_effort"] = (
                None if args.openai_reasoning_effort == "default"
                else args.openai_reasoning_effort
            )
        elif spec["type"] == "gemini":
            vlm_kwargs["gemini_thinking_level"] = (
                None if args.gemini_thinking_level == "default"
                else args.gemini_thinking_level
            )
        vlm = VLMModel(**vlm_kwargs)
        vlm.load()
        try:
            for level in args.levels:
                run_level(
                    vlm, items_by_id, manifest, model_dir, args.arm, level,
                    args.mode, visual_representation=args.visual_representation,
                )
        finally:
            vlm.unload()
        summarize(model_dir, args.arm, args.mode)


if __name__ == "__main__":
    main()
