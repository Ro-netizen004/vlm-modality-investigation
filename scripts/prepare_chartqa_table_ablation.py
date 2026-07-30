#!/usr/bin/env python3
"""Render plain-table images for the ChartQA representation ablation.

The table image preserves the official facts underlying the original chart.
The conflicting report and its designated answer are not edited: changing the
table's answer to the report answer would remove the conflict rather than
isolate chart-versus-table representation.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = (
    ROOT
    / "submission"
    / "anonymous_artifact"
    / "data"
    / "chartqa_conflict"
    / "items.jsonl"
)
DEFAULT_CHART_ROOT = DEFAULT_ITEMS.parent
DEFAULT_INSPECT = (
    ROOT
    / "outputs"
    / "chartqa_curation"
    / "chartqa_curation_workbook.xlsx.inspect.ndjson"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "chartqa_table_ablation" / "pilot"


class TableDoesNotFit(ValueError):
    """Raised when a complete table cannot fit the matched chart canvas."""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_source_tables(inspect_path: Path) -> dict[int, dict]:
    """Load the Source Tables sheet from the frozen workbook inspection."""
    with inspect_path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"kind":"table"' not in line or '"sheet":"Source Tables"' not in line:
                continue
            record = json.loads(line)
            values = record["values"]
            header = values[0]
            required = ["Pool ID", "Question", "Official table data"]
            missing = [name for name in required if name not in header]
            if missing:
                raise RuntimeError(f"Source Tables sheet is missing columns: {missing}")
            indices = {name: header.index(name) for name in required}
            rows = {}
            for row in values[1:]:
                if not row or row[indices["Pool ID"]] in (None, ""):
                    continue
                pool_id = int(row[indices["Pool ID"]])
                rows[pool_id] = {
                    "question": str(row[indices["Question"]]),
                    "official_table_data": str(row[indices["Official table data"]]),
                }
            return rows
    raise RuntimeError("Could not find the Source Tables sheet in the inspection file.")


def parse_table(value: str) -> list[list[str]]:
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(value))
        if any(cell.strip() for cell in row)
    ]
    if len(rows) < 2:
        raise ValueError("Official table must contain a header and at least one data row.")
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def text_width(draw: ImageDraw.ImageDraw, value: str, chosen_font) -> float:
    if not value:
        return 0.0
    box = draw.textbbox((0, 0), value, font=chosen_font)
    return float(box[2] - box[0])


def wrap_cell(draw, value: str, chosen_font, max_width: int) -> list[str]:
    value = str(value)
    if not value:
        return [""]
    words = value.split()
    if len(words) == 1 and text_width(draw, value, chosen_font) <= max_width:
        return [value]
    lines, current = [], ""
    for word in words or [value]:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, chosen_font) <= max_width:
            current = candidate
        elif current:
            lines.append(current)
            current = word
        else:
            # Hard-wrap a single long token.
            piece = ""
            for char in word:
                if piece and text_width(draw, piece + char, chosen_font) > max_width:
                    lines.append(piece)
                    piece = char
                else:
                    piece += char
            current = piece
    if current or not lines:
        lines.append(current)
    return lines


def column_widths(rows: list[list[str]], total_width: int) -> list[int]:
    columns = len(rows[0])
    weights = []
    for col in range(columns):
        longest = max(len(str(row[col])) for row in rows)
        weights.append(max(4.0, math.sqrt(longest + 1)))
    available = total_width - 2
    raw = [available * value / sum(weights) for value in weights]
    widths = [max(48, int(value)) for value in raw]
    # Correct rounding and minimum-width inflation while keeping the canvas fixed.
    while sum(widths) > available:
        largest = max(range(columns), key=lambda index: widths[index])
        if widths[largest] <= 48:
            break
        widths[largest] -= 1
    while sum(widths) < available:
        widths[sum(widths) % columns] += 1
    return widths


def layout(rows, canvas_size, size):
    width, height = canvas_size
    probe = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(probe)
    body_font, header_font = font(size), font(size, bold=True)
    widths = column_widths(rows, width - 12)
    padding_x, padding_y = max(3, size // 3), max(2, size // 4)
    line_height = int(size * 1.28)
    wrapped, heights = [], []
    for row_index, row in enumerate(rows):
        chosen = header_font if row_index == 0 else body_font
        wrapped_row = [
            wrap_cell(draw, cell, chosen, max(12, widths[col] - 2 * padding_x))
            for col, cell in enumerate(row)
        ]
        wrapped.append(wrapped_row)
        heights.append(
            max(len(lines) for lines in wrapped_row) * line_height + 2 * padding_y
        )
    return {
        "body_font": body_font,
        "header_font": header_font,
        "widths": widths,
        "wrapped": wrapped,
        "heights": heights,
        "padding_x": padding_x,
        "padding_y": padding_y,
        "line_height": line_height,
        "fits": sum(heights) <= height - 12,
    }


def render_table(rows: list[list[str]], canvas_size: tuple[int, int]) -> Image.Image:
    display_rows = [row.copy() for row in rows]
    if display_rows[0][0].strip().lower() == "characteristic":
        # The official tables use this generic field name extensively.  A shorter
        # synonymous label avoids splitting a single word in narrow multi-column
        # tables while leaving every factual value untouched.
        display_rows[0][0] = "Category"
    chosen = None
    for size in range(18, 7, -1):
        chosen = layout(display_rows, canvas_size, size)
        if chosen["fits"]:
            break
    if not chosen["fits"] and len(display_rows[0]) == 2:
        # Very long two-column tables can remain readable by continuing the
        # rows in a second side-by-side panel. Every fact is preserved, and the
        # repeated header makes the reading order explicit.
        header, body = display_rows[0], display_rows[1:]
        split = math.ceil(len(body) / 2)
        left, right = body[:split], body[split:]
        display_rows = [header + header]
        for index in range(split):
            right_row = right[index] if index < len(right) else ["", ""]
            display_rows.append(left[index] + right_row)
        for size in range(18, 7, -1):
            chosen = layout(display_rows, canvas_size, size)
            if chosen["fits"]:
                break
    if not chosen["fits"]:
        raise TableDoesNotFit(
            f"Table with {len(rows) - 1} data rows and {len(rows[0])} columns "
            f"does not fit canvas {canvas_size} at the minimum font size."
        )
    width, height = canvas_size
    image = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(image)
    table_height = sum(chosen["heights"])
    y = max(6, (height - table_height) // 2)
    for row_index, lines_by_cell in enumerate(chosen["wrapped"]):
        row_height = chosen["heights"][row_index]
        fill = "#E8EEF5" if row_index == 0 else ("#FFFFFF" if row_index % 2 else "#F7F7F7")
        x = 6
        for col, lines in enumerate(lines_by_cell):
            cell_width = chosen["widths"][col]
            draw.rectangle(
                (x, y, x + cell_width, y + row_height),
                fill=fill,
                outline="#707070",
                width=1,
            )
            chosen_font = (
                chosen["header_font"] if row_index == 0 else chosen["body_font"]
            )
            text_y = y + chosen["padding_y"]
            for line in lines:
                draw.text(
                    (x + chosen["padding_x"], text_y),
                    line,
                    fill="#111111",
                    font=chosen_font,
                )
                text_y += chosen["line_height"]
            x += cell_width
        y += row_height
    return image


def normalized(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--workbook-inspect", type=Path, default=DEFAULT_INSPECT)
    parser.add_argument("--chart-root", type=Path, default=DEFAULT_CHART_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--exclude-ids", nargs="*", type=int, default=[45])
    parser.add_argument(
        "--on-overflow",
        choices=("error", "skip"),
        default="error",
        help="Fail or record a prespecified exclusion when a complete table cannot fit.",
    )
    args = parser.parse_args()

    source_tables = load_source_tables(args.workbook_inspect)
    items = [
        row for row in read_jsonl(args.items)
        if int(row["conflict_id"]) not in set(args.exclude_ids)
    ]
    if args.limit is not None:
        items = items[: args.limit]

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest, skipped = [], []
    for item in items:
        pool_id = int(item["pool_conflict_id"])
        source = source_tables.get(pool_id)
        if source is None:
            raise RuntimeError(f"No Source Tables row for pool ID {pool_id}.")
        if source["question"].strip() != item["question"].strip():
            raise RuntimeError(f"Question mismatch for conflict {item['conflict_id']}.")
        rows = parse_table(source["official_table_data"])
        chart_path = args.chart_root / item["image_file"]
        with Image.open(chart_path) as chart:
            canvas_size = chart.size
        try:
            rendered = render_table(rows, canvas_size)
        except TableDoesNotFit as exc:
            if args.on_overflow == "error":
                raise
            skipped.append(
                {
                    "conflict_id": int(item["conflict_id"]),
                    "pool_conflict_id": pool_id,
                    "question": item["question"],
                    "reason": "complete_table_does_not_fit_matched_canvas",
                    "detail": str(exc),
                    "table_rows": len(rows) - 1,
                    "table_columns": len(rows[0]),
                    "table_canvas": list(canvas_size),
                }
            )
            continue
        output_path = image_dir / f"{int(item['conflict_id']):03d}.png"
        rendered.save(output_path, format="PNG")
        payload = output_path.read_bytes()
        flattened = " ".join(cell for row in rows for cell in row)
        manifest.append(
            {
                **item,
                "representation": "plain_table_image",
                "table_image_file": output_path.relative_to(args.output_dir).as_posix(),
                "table_image_sha256": hashlib.sha256(payload).hexdigest(),
                "table_canvas": list(canvas_size),
                "table_rows": len(rows) - 1,
                "table_columns": len(rows[0]),
                "chart_answer_string_present": (
                    normalized(item["chart_answer"]) in normalized(flattened)
                ),
                "official_table_data": source["official_table_data"],
            }
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    skipped_path = args.output_dir / "skipped.jsonl"
    with skipped_path.open("w", encoding="utf-8") as handle:
        for row in skipped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "items": len(manifest),
        "requested_items": len(items),
        "skipped_overflow": len(skipped),
        "excluded_conflict_ids": sorted(set(args.exclude_ids)),
        "all_questions_matched": True,
        "chart_answer_string_present": sum(
            bool(row["chart_answer_string_present"]) for row in manifest
        ),
        "manifest": manifest_path.name,
        "skipped_manifest": skipped_path.name,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Rendered tables -> {image_dir}")


if __name__ == "__main__":
    main()
