#!/usr/bin/env python3
"""Collate per-model legibility results into `legibility_all.json` (raw behavioral)
and `legibility_all_rescored.json` (reasoning-rescored) for the reorganized results
tree, where each arm is laid out as  <phase_root>/<benchmark>/<model>/level_*.json.

This is a layout-agnostic replacement for `run_legibility.py --merge`, whose path
logic assumes the old gsm8k-at-root / `text_legibility/` conventions and does not
match the current `<benchmark>/<model>/` layout.

Usage:
    python scripts/build_legibility_all.py results/phase6_legibility results/phase7_text_legibility
    # (defaults to those two if no args given)
"""
import json, os, re, sys

RAW_RE  = re.compile(r"^level_(\d+)_.*(?<!_rescored)\.json$")
RESC_RE = re.compile(r"^level_(\d+)_.*_rescored\.json$")


def _levels(model_dir, pat):
    """{level:int -> parsed json} for files in model_dir matching pattern."""
    out = {}
    for fn in os.listdir(model_dir):
        m = pat.match(fn)
        if not m:
            continue
        try:
            with open(os.path.join(model_dir, fn)) as f:
                out[int(m.group(1))] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def raw_summary(model, lv):
    o = sorted(lv)
    n = next((lv[L].get("n_problems") for L in o if lv[L].get("n_problems")), None)
    return {
        "model": model, "n": n, "levels_complete": o,
        "text_preference_by_level": {L: lv[L].get("text_preference") for L in o},
        "neither_rate_by_level":    {L: lv[L].get("neither_rate")    for L in o},
    }


def resc_summary(model, lv):
    o = sorted(lv)
    return {
        "model": model,
        "text_preference_by_level":     {L: lv[L].get("text_preference")     for L in o},
        "text_preference_raw_by_level": {L: lv[L].get("text_preference_raw") for L in o},
        "decidable_by_level":           {L: lv[L].get("decidable")           for L in o},
    }


def process(bench_dir):
    raw, resc = {}, {}
    for model in sorted(os.listdir(bench_dir)):
        md = os.path.join(bench_dir, model)
        if not os.path.isdir(md):
            continue
        lv = _levels(md, RAW_RE)
        if lv:
            raw[model] = raw_summary(model, lv)
        rv = _levels(md, RESC_RE)
        if rv:
            resc[model] = resc_summary(model, rv)
    if raw:
        with open(os.path.join(bench_dir, "legibility_all.json"), "w") as f:
            json.dump(raw, f, indent=2)
    if resc:
        with open(os.path.join(bench_dir, "legibility_all_rescored.json"), "w") as f:
            json.dump(resc, f, indent=2)
    return len(raw), len(resc)


def main(roots):
    for root in roots:
        if not os.path.isdir(root):
            print(f"skip (not found): {root}"); continue
        for bench in ("gsm8k", "svamp"):
            bd = os.path.join(root, bench)
            if not os.path.isdir(bd):
                continue
            nraw, nresc = process(bd)
            print(f"{bd}:  legibility_all.json ({nraw} models), "
                  f"legibility_all_rescored.json ({nresc} models)")


if __name__ == "__main__":
    args = sys.argv[1:] or ["results/phase6_legibility", "results/phase7_text_legibility"]
    main(args)
