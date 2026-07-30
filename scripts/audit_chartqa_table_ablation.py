#!/usr/bin/env python3
"""Audit source-table data -> frozen table metadata -> rendered PNG."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_chartqa_table_ablation import (
    DEFAULT_INSPECT,
    load_source_tables,
    parse_table,
    render_table,
)


DEFAULT_MANIFEST = (
    ROOT / "outputs" / "chartqa_table_ablation" / "full229" / "manifest.jsonl"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_source_csv(chartqa_root: Path, recorded_path: str) -> Path:
    relative = Path(recorded_path.replace("\\", "/"))
    candidates = [chartqa_root / relative]
    parts = relative.parts
    if parts and parts[0].lower() == "chartqa dataset":
        candidates.append(chartqa_root / Path(*parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve official source table {recorded_path!r}. Tried:\n{rendered}"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    manifest_path: Path,
    chartqa_root: Path | None = None,
    workbook_inspect: Path | None = None,
) -> dict:
    if (chartqa_root is None) == (workbook_inspect is None):
        raise ValueError("Provide exactly one of chartqa_root or workbook_inspect")
    rows = read_jsonl(manifest_path)
    workbook_tables = (
        load_source_tables(workbook_inspect) if workbook_inspect else None
    )
    errors = []
    source_hashes = {}
    rendered_verified = 0
    for row in rows:
        conflict_id = int(row["conflict_id"])
        try:
            if chartqa_root is not None:
                source_path = resolve_source_csv(chartqa_root, row["source_table"])
                source_text = source_path.read_text(encoding="utf-8-sig")
                source_bytes = source_path.read_bytes()
            else:
                source = workbook_tables.get(int(row["pool_conflict_id"]))
                if source is None:
                    raise ValueError("pool_conflict_id is absent from Source Tables")
                if source["question"].strip() != row["question"].strip():
                    raise ValueError("workbook question differs from manifest question")
                source_text = source["official_table_data"]
                source_bytes = source_text.encode("utf-8")
            official_rows = parse_table(source_text)
            frozen_rows = parse_table(row["official_table_data"])
            if official_rows != frozen_rows:
                raise ValueError("source-table cells differ from official_table_data")

            image_path = (
                manifest_path.parent / row["table_image_file"]
            ).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"missing rendered table image: {image_path}")
            if sha256(image_path) != row["table_image_sha256"]:
                raise ValueError("rendered table image checksum mismatch")

            canvas = tuple(int(value) for value in row["table_canvas"])
            rerendered = render_table(official_rows, canvas)
            payload = rerendered.tobytes()
            # PNG encoding is deterministic for the same Pillow build in the
            # preparation environment, but pixel equality is the portable test.
            from PIL import Image
            with Image.open(image_path) as saved:
                if saved.convert("RGB").tobytes() != payload:
                    raise ValueError("rendered PNG pixels differ from official CSV render")

            source_hashes[str(conflict_id)] = hashlib.sha256(source_bytes).hexdigest()
            rendered_verified += 1
        except Exception as error:
            errors.append({
                "conflict_id": conflict_id,
                "source_table": row.get("source_table"),
                "error": str(error),
            })

    return {
        "manifest": str(manifest_path),
        "provenance_scope": (
            "official_chartqa_csv" if chartqa_root is not None
            else "frozen_curation_workbook"
        ),
        "official_csv_verified": chartqa_root is not None,
        "chartqa_root": str(chartqa_root.resolve()) if chartqa_root else None,
        "workbook_inspect": (
            str(workbook_inspect.resolve()) if workbook_inspect else None
        ),
        "items": len(rows),
        "source_table_matches": len(rows) - len(errors),
        "rendered_images_verified": rendered_verified,
        "errors": errors,
        "source_table_sha256_by_conflict_id": source_hashes,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--chartqa-root",
        type=Path,
        help=(
            "Extracted official ChartQA root, either the directory containing "
            "'ChartQA Dataset/' or that directory itself."
        ),
    )
    source.add_argument(
        "--workbook-inspect",
        type=Path,
        nargs="?",
        const=DEFAULT_INSPECT,
        help=(
            "Audit against the frozen curation-workbook inspection. If the path "
            "is omitted, use the repository's canonical inspection file."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "chartqa_table_ablation" / "audit.json",
    )
    args = parser.parse_args()

    result = audit(
        args.manifest,
        chartqa_root=args.chartqa_root,
        workbook_inspect=args.workbook_inspect,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Audited {result['items']} items: "
        f"source_table_matches={result['source_table_matches']} "
        f"rendered_images_verified={result['rendered_images_verified']}"
    )
    print(f"Audit report -> {args.output}")
    if not result["passed"]:
        preview = "\n".join(
            f"  conflict {row['conflict_id']}: {row['error']}"
            for row in result["errors"][:10]
        )
        raise SystemExit(f"Audit FAILED with {len(result['errors'])} errors:\n{preview}")
    print("Audit PASSED")


if __name__ == "__main__":
    main()
