"""Export and compile reviewed evidence-bearing ChartQA conflict reports.

The official ChartQA repository is required because the lmms-lab evaluation copy
omits underlying CSV tables. Workflow:
  1. export: join an assertion draft manifest to official QA records/tables and
     create a TSV review sheet;
  2. humans (or a generator followed by humans) fill text_report, optionally edit
     text_answer, and mark entailed=yes with reviewer initials;
  3. compile: validate every row and emit the evidence manifest consumed by
     run_chartqa_conflict.py --report-type evidence.
"""

import argparse
import csv
import io
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


FIELDS = [
    "conflict_id", "dataset_index", "question", "image_answer", "text_answer",
    "answer_type", "image_label", "text_label", "source_table", "table_data",
    "chart_value_candidates", "counterfactual_strategy", "unit_class",
    "text_report", "entailed", "counterfactual_valid", "reviewer", "status",
    "exclusion_reason", "notes",
]

ALLOWED_STRATEGIES = {
    "chart_value", "nearby_category_value", "rank_swap",
    "arithmetic_alternative", "unit_preserving_perturbation", "boolean_flip",
}
NUMERIC_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?\s*%?")


def normalized_answer(value):
    text = str(value).strip().lower().rstrip(".")
    if text in {"yes", "no"}:
        return ("boolean", text, None)
    match = re.fullmatch(r"(-?[\d,]+(?:\.\d+)?)\s*(%)?", text)
    if not match:
        return ("text", re.sub(r"\s+", " ", text), None)
    try:
        number = Decimal(match.group(1).replace(",", "")).normalize()
    except InvalidOperation:
        return ("text", text, None)
    return ("numeric", number, "percent" if match.group(2) else "number")


def chart_numeric_candidates(table_text, gold):
    gold_norm = normalized_answer(gold)
    values = []
    # Parse cells first: applying a numeric regex to raw CSV text can mistake a
    # delimiter for a thousands separator (e.g. cells "2021","3.89" -> "2021,3.89").
    for csv_row in csv.reader(io.StringIO(table_text)):
        for cell in csv_row:
            for match in NUMERIC_RE.findall(cell):
                candidate = match.strip()
                norm = normalized_answer(candidate)
                if norm[0] != "numeric" or norm == gold_norm:
                    continue
                if gold_norm[0] == "numeric" and norm[2] != gold_norm[2]:
                    continue
                if candidate not in values:
                    values.append(candidate)
    return values[:30]


def validate_counterfactual(row, index):
    errors = []
    gold = row.get("image_answer", "").strip()
    alternate = row.get("text_answer", "").strip()
    gold_norm, alternate_norm = normalized_answer(gold), normalized_answer(alternate)
    strategy = row.get("counterfactual_strategy", "").strip()
    unit_class = row.get("unit_class", "").strip()
    if strategy not in ALLOWED_STRATEGIES:
        errors.append(f"row {index}: invalid counterfactual_strategy {strategy!r}")
    if gold_norm == alternate_norm:
        errors.append(f"row {index}: gold and counterfactual normalize to the same answer")
    if gold_norm[0] != alternate_norm[0]:
        errors.append(f"row {index}: answer types differ ({gold_norm[0]} vs {alternate_norm[0]})")
    if gold_norm[0] == "numeric" and gold_norm[2] != alternate_norm[2]:
        errors.append(f"row {index}: percentage/raw-count units are mixed")
    if gold_norm[0] == "numeric" and not unit_class:
        errors.append(f"row {index}: numeric/date answer requires unit_class")
    if gold_norm[0] == "boolean":
        if strategy != "boolean_flip" or {gold_norm[1], alternate_norm[1]} != {"yes", "no"}:
            errors.append(f"row {index}: yes/no counterfactual must use boolean_flip")
    gold_text, alternate_text = gold.lower().strip(), alternate.lower().strip()
    if gold_text and alternate_text and (gold_text in alternate_text or alternate_text in gold_text):
        errors.append(f"row {index}: one answer is a substring of the other")
    if row.get("counterfactual_valid", "").strip().lower() not in {"yes", "true", "1"}:
        errors.append(f"row {index}: counterfactual not marked valid/coherent")
    return errors


def load_official(chartqa_root: Path):
    records = {}
    for name in ("test_human.json", "test_augmented.json"):
        matches = list(chartqa_root.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Could not find {name} below {chartqa_root}")
        with matches[0].open(encoding="utf-8") as handle:
            for row in json.load(handle):
                key = (str(row.get("query", "")).strip(), str(row.get("label", "")).strip())
                records.setdefault(key, []).append((row, matches[0]))
    return records


def find_table(json_path: Path, imgname: str):
    split_root = json_path.parent
    candidate = split_root / "tables" / f"{Path(imgname).stem}.csv"
    if candidate.exists():
        return candidate
    matches = list(split_root.rglob(f"{Path(imgname).stem}.csv"))
    return matches[0] if matches else None


def export_sheet(draft_path: Path, chartqa_root: Path, output: Path):
    with draft_path.open(encoding="utf-8") as handle:
        draft = json.load(handle)
    official = load_official(chartqa_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in draft:
            key = (str(row["question"]).strip(), str(row["image_answer"]).strip())
            candidates = official.get(key, [])
            if not candidates:
                raise RuntimeError(f"No official ChartQA match for: {key}")
            official_row, json_path = candidates[0]
            table = find_table(json_path, official_row["imgname"])
            if table is None:
                raise RuntimeError(f"No table for {official_row['imgname']}")
            relative = table.relative_to(chartqa_root)
            writer.writerow({
                **{field: row.get(field, "") for field in FIELDS},
                "source_table": str(relative).replace("\\", "/"),
                "table_data": table.read_text(encoding="utf-8", errors="replace"),
                "chart_value_candidates": " | ".join(chart_numeric_candidates(
                    table.read_text(encoding="utf-8", errors="replace"), row["image_answer"]
                )),
                "counterfactual_strategy": "boolean_flip" if row["answer_type"] == "boolean" else "",
                "unit_class": "percent" if str(row["image_answer"]).strip().endswith("%") else "",
                "text_report": "",
                "entailed": "",
                "counterfactual_valid": "",
                "reviewer": "",
                "status": "reserve",
                "exclusion_reason": "",
                "notes": "",
            })
    print(f"Exported {len(draft)} rows -> {output}")


def apply_updates(sheet: Path, updates_path: Path):
    with sheet.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with updates_path.open(encoding="utf-8") as handle:
        updates = {int(row["conflict_id"]): row for row in json.load(handle)}
    seen = set()
    for row in rows:
        conflict_id = int(row["conflict_id"])
        if conflict_id in updates:
            for key, value in updates[conflict_id].items():
                if key != "conflict_id":
                    if key not in FIELDS:
                        raise RuntimeError(f"Unknown curation field {key!r}")
                    row[key] = value
            seen.add(conflict_id)
    missing = set(updates) - seen
    if missing:
        raise RuntimeError(f"Updates reference missing conflict IDs: {sorted(missing)}")
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Applied {len(updates)} reviewed updates -> {sheet}")


def compile_manifest(sheet: Path, output: Path, target_size: int,
                     allow_undocumented_exclusions: bool = False):
    compiled, errors = [], []
    with sheet.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for index, row in enumerate(rows):
        status = row.get("status", "").strip().lower()
        if status == "exclude":
            if (not allow_undocumented_exclusions and
                    not row.get("exclusion_reason", "").strip()):
                errors.append(f"row {index}: excluded without an exclusion_reason")
            continue
        if status != "include":
            continue
        report = row.get("text_report", "").strip()
        answer = row.get("text_answer", "").strip()
        entailed = row.get("entailed", "").strip().lower() in {"yes", "true", "1"}
        reviewer = row.get("reviewer", "").strip()
        if not report:
            errors.append(f"row {index}: empty text_report")
        if not answer:
            errors.append(f"row {index}: empty text_answer")
        if not entailed:
            errors.append(f"row {index}: not marked entailed")
        if not reviewer:
            errors.append(f"row {index}: missing reviewer")
        if report.lower().strip().startswith(("the answer is", "answer:")):
            errors.append(f"row {index}: report appears answer-asserting, not evidence-bearing")
        errors.extend(validate_counterfactual(row, index))
        compiled.append({
            "conflict_id": int(row["conflict_id"]),
            "pool_conflict_id": int(row["conflict_id"]),
            "dataset_index": int(row["dataset_index"]),
            "question": row["question"],
            "image_answer": row["image_answer"],
            "text_answer": answer,
            "answer_type": row["answer_type"],
            "image_label": row["image_label"],
            "text_label": row["text_label"],
            "report_type": "evidence",
            "counterfactual_strategy": row["counterfactual_strategy"].strip(),
            "unit_class": row.get("unit_class", "").strip(),
            "text_report": report,
            "source_table": row["source_table"],
            "evidence_validation": {"entailed": True, "counterfactual_valid": True,
                                    "reviewer": reviewer,
                                    "notes": row.get("notes", "")},
        })
    if errors:
        raise RuntimeError("Curation sheet failed validation:\n" + "\n".join(errors[:50]))
    compiled.sort(key=lambda row: row["pool_conflict_id"])
    if len(compiled) != target_size:
        raise RuntimeError(f"Expected exactly {target_size} included rows; found {len(compiled)}")
    for conflict_id, row in enumerate(compiled):
        row["conflict_id"] = conflict_id
        row["image_label"], row["text_label"] = (
            ("A", "B") if conflict_id % 2 == 0 else ("B", "A")
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(compiled, handle, indent=2, ensure_ascii=False)
    print(f"Compiled {len(compiled)} reviewed rows -> {output}")


def compile_workbook_export(export_path: Path, output: Path, target_size: int,
                            exclude_ids: set[int]):
    """Compile the JSON exported from the curation workbook.

    Dataset indices are resolved against the public ChartQA test metadata on the
    machine doing the compilation. The workbook remains unchanged.
    """
    from datasets import load_dataset

    with export_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    selected = [
        row for row in rows
        if str(row.get("status", "")).strip().lower() == "include"
        and int(row["conflict_id"]) not in exclude_ids
    ]
    errors = []
    for index, row in enumerate(selected):
        if not str(row.get("text_report", "")).strip():
            errors.append(f"row {index}: empty text_report")
        if not str(row.get("text_answer", "")).strip():
            errors.append(f"row {index}: empty text_answer")
        if str(row.get("entailed", "")).strip().lower() not in {"yes", "true", "1"}:
            errors.append(f"row {index}: not marked entailed")
        if str(row.get("counterfactual_valid", "")).strip().lower() not in {"yes", "true", "1"}:
            errors.append(f"row {index}: counterfactual not marked valid")
        if not str(row.get("reviewer", "")).strip():
            errors.append(f"row {index}: missing reviewer")
        errors.extend(validate_counterfactual(row, index))
    if len(selected) != target_size:
        errors.append(f"expected {target_size} included rows after exclusions; found {len(selected)}")
    if errors:
        raise RuntimeError("Workbook export failed validation:\n" + "\n".join(errors[:50]))

    metadata = load_dataset("lmms-lab/ChartQA", split="test")
    metadata_without_images = (
        metadata.remove_columns("image") if "image" in metadata.column_names else metadata
    )
    by_key = {}
    for dataset_index, item in enumerate(metadata_without_images):
        question = str(item.get("question", item.get("query", ""))).strip()
        answer = str(item.get("answer", item.get("label", ""))).strip()
        by_key.setdefault((question, answer), []).append(dataset_index)

    compiled = []
    for row in selected:
        key = (str(row["question"]).strip(), str(row["image_answer"]).strip())
        matches = by_key.get(key, [])
        if len(matches) > 1 and "image" in metadata.column_names:
            # ChartQA occasionally duplicates a QA record at adjacent test indices.
            # Resolve it only when the underlying chart pixels are identical.
            import hashlib
            import io

            hashes = set()
            for match in matches:
                image = metadata[match]["image"].convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                hashes.add(hashlib.sha256(buffer.getvalue()).hexdigest())
            if len(hashes) == 1:
                matches = [min(matches)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one ChartQA test match for pool ID {row['conflict_id']} "
                f"{key!r}; found {len(matches)}"
            )
        compiled.append({
            "conflict_id": len(compiled),
            "pool_conflict_id": int(row["conflict_id"]),
            "dataset_index": matches[0],
            "question": row["question"],
            "image_answer": row["image_answer"],
            "text_answer": row["text_answer"],
            "answer_type": row["answer_type"],
            "image_label": "A" if len(compiled) % 2 == 0 else "B",
            "text_label": "B" if len(compiled) % 2 == 0 else "A",
            "report_type": "evidence",
            "counterfactual_strategy": row["counterfactual_strategy"],
            "unit_class": row.get("unit_class", ""),
            "text_report": row["text_report"],
            "source_table": row["source_table"],
            "evidence_validation": {
                "entailed": True,
                "counterfactual_valid": True,
                "reviewer": row["reviewer"],
                "notes": row.get("notes", ""),
            },
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(compiled, handle, indent=2, ensure_ascii=False)
    print(f"Compiled {len(compiled)} workbook rows -> {output}")
    print(f"Excluded pool IDs: {sorted(exclude_ids)}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--draft-manifest", type=Path, required=True)
    export.add_argument("--chartqa-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    compile_cmd = subparsers.add_parser("compile")
    compile_cmd.add_argument("--sheet", type=Path, required=True)
    compile_cmd.add_argument("--output", type=Path, required=True)
    compile_cmd.add_argument("--target-size", type=int, default=300)
    compile_cmd.add_argument("--allow-undocumented-exclusions", action="store_true",
                             help="Smoke-test only: do not block on excluded reserve-pool "
                                  "rows whose exclusion reason is not yet documented")
    apply_cmd = subparsers.add_parser("apply")
    apply_cmd.add_argument("--sheet", type=Path, required=True)
    apply_cmd.add_argument("--updates", type=Path, required=True)
    workbook_cmd = subparsers.add_parser("compile-workbook-export")
    workbook_cmd.add_argument("--export", type=Path, required=True)
    workbook_cmd.add_argument("--output", type=Path, required=True)
    workbook_cmd.add_argument("--target-size", type=int, required=True)
    workbook_cmd.add_argument("--exclude-ids", nargs="+", type=int, default=[])
    args = parser.parse_args()
    if args.command == "export":
        export_sheet(args.draft_manifest, args.chartqa_root, args.output)
    elif args.command == "apply":
        apply_updates(args.sheet, args.updates)
    elif args.command == "compile-workbook-export":
        compile_workbook_export(args.export, args.output, args.target_size,
                                set(args.exclude_ids))
    else:
        compile_manifest(args.sheet, args.output, args.target_size,
                         args.allow_undocumented_exclusions)


if __name__ == "__main__":
    main()
