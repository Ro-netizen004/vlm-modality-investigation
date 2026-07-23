#!/usr/bin/env python3
"""Audit L0 ChartQA conflict smoke outputs before launching degraded endpoints."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["i"])] = row
    return rows


def find_level(model_dir, mode):
    matches = list(model_dir.glob(f"level_0_*.{mode}.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one L0 {mode} file in {model_dir}; found {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Condition root containing evidence/{image,text}/<model>")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--expected", type=int, default=35)
    args = parser.parse_args()

    failed = False
    for model in args.models:
        generation, cll = {}, {}
        for arm in ("image", "text"):
            model_dir = args.root / "evidence" / arm / model
            generation[arm] = load(find_level(model_dir, "generation"))
            cll[arm] = load(find_level(model_dir, "cll"))
            counts = Counter(row.get("follows", "invalid") for row in generation[arm].values())
            valid_cll = sum((row.get("margin") or {}).get("margin_mean") is not None
                            for row in cll[arm].values())
            errors = sum("error" in row for row in generation[arm].values())
            print(f"{model} {arm}: generation={len(generation[arm])}/{args.expected} "
                  f"counts={dict(counts)} errors={errors}; "
                  f"CLL={valid_cll}/{args.expected}")
            failed |= len(generation[arm]) != args.expected or valid_cll != args.expected or errors > 0

        shared = sorted(generation["image"].keys() & generation["text"].keys())
        generation_match = sum(
            generation["image"][i].get("prediction") == generation["text"][i].get("prediction")
            for i in shared
        )
        cll_shared = sorted(cll["image"].keys() & cll["text"].keys())
        cll_match = sum(
            (cll["image"][i].get("margin") or {}).get("margin_mean") ==
            (cll["text"][i].get("margin") or {}).get("margin_mean")
            for i in cll_shared
        )
        decidable = agreement = 0
        for i, row in generation["image"].items():
            follows = row.get("follows")
            margin = (cll["image"].get(i, {}).get("margin") or {}).get("margin_mean")
            if follows in {"image", "text"} and margin is not None and margin != 0:
                decidable += 1
                agreement += (margin > 0) == (follows == "text")
        print(f"{model} L0 arm invariance: generation={generation_match}/{len(shared)}, "
              f"CLL={cll_match}/{len(cll_shared)}; sign agreement="
              f"{agreement}/{decidable if decidable else 0}")
        failed |= generation_match != len(shared) or cll_match != len(cll_shared)

    if failed:
        raise SystemExit("Smoke audit FAILED; inspect outputs before launching L5")
    print("Smoke audit PASSED")


if __name__ == "__main__":
    main()
