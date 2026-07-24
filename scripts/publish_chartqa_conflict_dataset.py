#!/usr/bin/env python3
"""Build and optionally publish the frozen ChartQA evidence-conflict dataset.

The source of truth is the compiled manifest, not the curation workbook. The
publisher revalidates every ChartQA index/question/answer join, embeds the
original chart image, records the manifest hash on every row, and creates the
Hub dataset as private unless --public is explicitly requested.
"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Value, load_dataset
from huggingface_hub import HfApi, whoami


DEFAULT_REPO = "vlm-modality-research/chartqa-evidence-conflict-v1"
DEFAULT_MANIFEST = Path(
    "results/phase_control/chartqa_conflict/evidence/manifest_230.json"
)
FROZEN_MANIFEST_SHA256 = (
    "388bce0572487024f5ac12261621cbab8931ec3032d8bf0c65a258c134d20842"
)
DESIGN_VERSION = "chartqa-same-question-conflict-v4"


def manifest_digest(rows):
    payload = json.dumps(
        rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/chartqa_evidence_conflict_v1"))
    parser.add_argument("--expected-size", type=int, default=230)
    parser.add_argument("--expected-sha256", default=FROZEN_MANIFEST_SHA256)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--public", action="store_true",
                        help="publish publicly; default is private")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    digest = manifest_digest(manifest)
    if len(manifest) != args.expected_size:
        raise RuntimeError(
            f"Manifest has {len(manifest)} rows; expected {args.expected_size}"
        )
    if digest != args.expected_sha256:
        raise RuntimeError(
            f"Manifest hash changed: {digest}; expected {args.expected_sha256}"
        )

    source = load_dataset("lmms-lab/ChartQA", split="test")
    rows = []
    for expected_conflict_id, item in enumerate(manifest):
        if int(item["conflict_id"]) != expected_conflict_id:
            raise RuntimeError(
                f"Non-contiguous conflict_id at row {expected_conflict_id}"
            )
        dataset_index = int(item["dataset_index"])
        official = source[dataset_index]
        question = str(official.get("question", official.get("query", ""))).strip()
        answer = str(official.get("answer", official.get("label", ""))).strip()
        if question != str(item["question"]).strip():
            raise RuntimeError(
                f"Question mismatch at conflict_id {expected_conflict_id}"
            )
        if answer != str(item["image_answer"]).strip():
            raise RuntimeError(
                f"Chart answer mismatch at conflict_id {expected_conflict_id}"
            )
        validation = item.get("evidence_validation", {})
        if not validation.get("entailed") or not validation.get("counterfactual_valid"):
            raise RuntimeError(
                f"Unvalidated report at conflict_id {expected_conflict_id}"
            )
        rows.append({
            "conflict_id": expected_conflict_id,
            "pool_conflict_id": int(item["pool_conflict_id"]),
            "chartqa_test_index": dataset_index,
            "image": official["image"].convert("RGB"),
            "question": item["question"],
            "chart_answer": str(item["image_answer"]),
            "report_answer": str(item["text_answer"]),
            "answer_type": item["answer_type"],
            "unit_class": item.get("unit_class", ""),
            "counterfactual_strategy": item["counterfactual_strategy"],
            "text_report": item["text_report"],
            "chart_source_label": item["image_label"],
            "report_source_label": item["text_label"],
            "source_table": item["source_table"],
            "reviewer": validation.get("reviewer", ""),
            "review_notes": validation.get("notes", ""),
            "manifest_sha256": digest,
            "design_version": DESIGN_VERSION,
            "source_dataset": "lmms-lab/ChartQA",
            "source_split": "test",
        })

    features = Features({
        "conflict_id": Value("int32"),
        "pool_conflict_id": Value("int32"),
        "chartqa_test_index": Value("int32"),
        "image": Image(),
        "question": Value("string"),
        "chart_answer": Value("string"),
        "report_answer": Value("string"),
        "answer_type": Value("string"),
        "unit_class": Value("string"),
        "counterfactual_strategy": Value("string"),
        "text_report": Value("string"),
        "chart_source_label": Value("string"),
        "report_source_label": Value("string"),
        "source_table": Value("string"),
        "reviewer": Value("string"),
        "review_notes": Value("string"),
        "manifest_sha256": Value("string"),
        "design_version": Value("string"),
        "source_dataset": Value("string"),
        "source_split": Value("string"),
    })
    dataset = Dataset.from_list(rows, features=features)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    print(
        f"Built {len(dataset)} rows -> {args.output_dir}\n"
        f"manifest_sha256={digest}"
    )

    if args.build_only:
        return
    try:
        identity = whoami()
    except Exception as error:
        raise SystemExit(
            "Not authenticated. Run `huggingface-cli login` and retry."
        ) from error
    print(f"Authenticated as {identity.get('name')}")
    dataset.push_to_hub(args.repo, split="test", private=not args.public)

    card = Path(__file__).resolve().parents[1] / "docs" / \
        "CHARTQA_CONFLICT_DATASET_CARD.md"
    HfApi().upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    info = HfApi().dataset_info(args.repo)
    print(f"Published: https://huggingface.co/datasets/{args.repo}")
    print(f"Pinned revision: {info.sha}")


if __name__ == "__main__":
    main()
