#!/usr/bin/env python3
"""McNemar test for paired accuracy between two experiment conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_benchmark.stats import format_mcnemar_report, mcnemar_from_csv


def parse_args():
    p = argparse.ArgumentParser(
        description="McNemar test on paired correct/incorrect outcomes (same problem_ids)."
    )
    p.add_argument(
        "csv_a",
        help="First results CSV (or single wide CSV if --csv-b omitted)",
    )
    p.add_argument(
        "--col-a",
        required=True,
        help="Correctness column for condition A (e.g. correct, correct_text)",
    )
    p.add_argument(
        "--csv-b",
        default=None,
        help="Second results CSV; omit to compare two columns in csv_a",
    )
    p.add_argument(
        "--col-b",
        default=None,
        help="Correctness column for condition B (required if --csv-b omitted)",
    )
    p.add_argument("--label-a", default=None, help="Display name for condition A")
    p.add_argument("--label-b", default=None, help="Display name for condition B")
    p.add_argument(
        "--id-col",
        default="problem_id",
        help="Join key column (default: problem_id)",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON results",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.csv_b is None and args.col_b is None:
        p.error("Provide --col-b when using a single CSV with two columns.")

    result = mcnemar_from_csv(
        args.csv_a,
        args.col_a,
        csv_b=args.csv_b,
        col_b=args.col_b,
        label_a=args.label_a,
        label_b=args.label_b,
        id_col=args.id_col,
    )
    print(format_mcnemar_report(result))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    main()
