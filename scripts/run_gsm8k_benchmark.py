#!/usr/bin/env python3
"""Deprecated alias — forwards to scripts/run_benchmark.py (src/ pipeline).

For the old vlm_benchmark CLI (--dataset-type, --mode), see vlm_benchmark/README.md.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

warnings.warn(
    "run_gsm8k_benchmark.py forwards to run_benchmark.py (src/). "
    "Legacy vlm_benchmark: see vlm_benchmark/README.md and docs/GETTING_STARTED.md",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "run_benchmark.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
