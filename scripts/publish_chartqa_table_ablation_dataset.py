#!/usr/bin/env python3
"""Build/publish ChartQA-Conflict with matched chart and table images."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi, whoami
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "outputs" / "chartqa_table_ablation" / "full229" / "manifest.jsonl"
)
DEFAULT_CHART_ROOT = (
    ROOT / "submission" / "anonymous_artifact" / "data" / "chartqa_conflict"
)
DEFAULT_REPO = "vlm-modality-research/chartqa-evidence-conflict-table-v1"
DEFAULT_AUDIT = ROOT / "outputs" / "chartqa_table_ablation" / "audit.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chart-root", type=Path, default=DEFAULT_CHART_ROOT)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "chartqa_evidence_conflict_table_v1",
    )
    parser.add_argument("--expected-size", type=int, default=229)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    if len(manifest) != args.expected_size:
        raise RuntimeError(
            f"Manifest has {len(manifest)} rows; expected {args.expected_size}"
        )
    if not args.audit_report.is_file():
        raise RuntimeError(
            f"Missing provenance audit {args.audit_report}. Run "
            "audit_chartqa_table_ablation.py first."
        )
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if not audit.get("passed") or int(audit.get("items", -1)) != len(manifest):
        raise RuntimeError(
            f"Provenance audit did not pass for all {len(manifest)} items: "
            f"{args.audit_report}"
        )
    audit_sha256 = hashlib.sha256(args.audit_report.read_bytes()).hexdigest()
    source_hashes = audit["source_table_sha256_by_conflict_id"]

    rows = []
    for item in manifest:
        conflict_id = int(item["conflict_id"])
        chart_path = (args.chart_root / item["image_file"]).resolve()
        table_path = (args.manifest.parent / item["table_image_file"]).resolve()
        if not chart_path.is_file():
            raise FileNotFoundError(f"Missing chart image: {chart_path}")
        if not table_path.is_file():
            raise FileNotFoundError(f"Missing table image: {table_path}")
        with PILImage.open(chart_path) as chart:
            chart_image = chart.convert("RGB").copy()
        with PILImage.open(table_path) as table:
            table_image = table.convert("RGB").copy()
        rows.append({
            "conflict_id": conflict_id,
            "pool_conflict_id": int(item["pool_conflict_id"]),
            "chartqa_test_index": int(item["chartqa_test_index"]),
            "chart_image": chart_image,
            "table_image": table_image,
            "question": item["question"],
            "chart_answer": str(item["chart_answer"]),
            "report_answer": str(item["report_answer"]),
            "text_report": item["text_report"],
            "answer_type": item["answer_type"],
            "unit_class": item["unit_class"],
            "counterfactual_strategy": item["counterfactual_strategy"],
            "chart_source_label": item["chart_source_label"],
            "report_source_label": item["report_source_label"],
            "source_table": item["source_table"],
            "official_table_data": item["official_table_data"],
            "table_image_sha256": item["table_image_sha256"],
            "source_table_sha256": source_hashes[str(conflict_id)],
            "provenance_scope": audit["provenance_scope"],
            "provenance_audit_sha256": audit_sha256,
            "original_manifest_sha256": item["manifest_sha256"],
            "design_version": "chartqa-chart-vs-table-ablation-v1",
            "source_dataset": "lmms-lab/ChartQA",
            "source_split": "test",
        })

    features = Features({
        "conflict_id": Value("int32"),
        "pool_conflict_id": Value("int32"),
        "chartqa_test_index": Value("int32"),
        "chart_image": Image(),
        "table_image": Image(),
        "question": Value("string"),
        "chart_answer": Value("string"),
        "report_answer": Value("string"),
        "text_report": Value("string"),
        "answer_type": Value("string"),
        "unit_class": Value("string"),
        "counterfactual_strategy": Value("string"),
        "chart_source_label": Value("string"),
        "report_source_label": Value("string"),
        "source_table": Value("string"),
        "official_table_data": Value("string"),
        "table_image_sha256": Value("string"),
        "source_table_sha256": Value("string"),
        "provenance_scope": Value("string"),
        "provenance_audit_sha256": Value("string"),
        "original_manifest_sha256": Value("string"),
        "design_version": Value("string"),
        "source_dataset": Value("string"),
        "source_split": Value("string"),
    })
    dataset = Dataset.from_list(rows, features=features)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    print(f"Built {len(dataset)} paired chart/table rows -> {args.output_dir}")

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
    card = ROOT / "docs" / "CHARTQA_TABLE_ABLATION_DATASET_CARD.md"
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
