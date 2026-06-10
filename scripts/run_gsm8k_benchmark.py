#!/usr/bin/env python3
"""Deprecated alias — use scripts/run_benchmark.py instead."""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

warnings.warn(
    "scripts/run_gsm8k_benchmark.py is deprecated; use scripts/run_benchmark.py",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "run_benchmark.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
