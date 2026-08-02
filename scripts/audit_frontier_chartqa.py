#!/usr/bin/env python3
"""Gate frontier ChartQA smoke tests before paying for a full run."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--expected", type=int, default=30)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 5])
    parser.add_argument("--max-empty-rate", type=float, default=0.05)
    parser.add_argument("--max-invalid-rate", type=float, default=0.25)
    parser.add_argument("--min-decidable-rate", type=float, default=0.50)
    parser.add_argument(
        "--max-format-violation-rate",
        type=float,
        default=0.05,
        help="Maximum fraction not matching one exact '#### <answer>' line.",
    )
    args = parser.parse_args()

    failed = False
    for model in args.models:
        for arm in ("image", "text"):
            model_dir = args.root / "evidence" / arm / model
            for level in args.levels:
                matches = list(model_dir.glob(f"level_{level}_*.generation.jsonl"))
                if len(matches) != 1:
                    print(f"FAIL {model} {arm} L{level}: found {len(matches)} files")
                    failed = True
                    continue
                rows = load_jsonl(matches[0])
                counts = Counter(row.get("follows", "invalid") for row in rows)
                empty = sum(not str(row.get("prediction", "")).strip() for row in rows)
                format_violations = sum(
                    re.fullmatch(r"####\s*[^\r\n]+", str(row.get("prediction", "")).strip())
                    is None
                    for row in rows
                )
                decidable = counts["image"] + counts["text"]
                finish = Counter(
                    str((row.get("api_response_meta") or {}).get("finish_reason"))
                    for row in rows
                )
                n = len(rows)
                empty_rate = empty / n if n else 1.0
                invalid_rate = counts["invalid"] / n if n else 1.0
                decidable_rate = decidable / n if n else 0.0
                format_violation_rate = format_violations / n if n else 1.0
                print(
                    f"{model} {arm} L{level}: n={n}/{args.expected} "
                    f"counts={dict(counts)} empty={empty_rate:.1%} "
                    f"invalid={invalid_rate:.1%} decidable={decidable_rate:.1%} "
                    f"format_violations={format_violation_rate:.1%} "
                    f"finish={dict(finish)}"
                )
                failed |= (
                    n != args.expected
                    or empty_rate > args.max_empty_rate
                    or invalid_rate > args.max_invalid_rate
                    or decidable_rate < args.min_decidable_rate
                    or format_violation_rate > args.max_format_violation_rate
                )

    if failed:
        raise SystemExit(
            "Frontier smoke gate FAILED. Do not launch the full run; inspect raw "
            "predictions and api_response_meta."
        )
    print("Frontier smoke gate PASSED.")


if __name__ == "__main__":
    main()
