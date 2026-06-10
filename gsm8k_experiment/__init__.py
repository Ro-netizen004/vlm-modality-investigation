"""
Deprecated package name — use ``vlm_benchmark`` instead.

This shim keeps ``import gsm8k_experiment`` and submodule imports working.
"""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Package gsm8k_experiment was renamed to vlm_benchmark. "
    "Update imports to use vlm_benchmark.",
    DeprecationWarning,
    stacklevel=2,
)

_PKG = "vlm_benchmark"
_TOP = (
    "config",
    "types",
    "data",
    "answer_parsing",
    "eval",
    "visualize",
    "stats",
    "run_experiment",
    "datasets",
    "models",
    "experiments",
)

for _name in _TOP:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"{_PKG}.{_name}")

for _name in ("gsm8k", "svamp", "base"):
    sys.modules[f"{__name__}.datasets.{_name}"] = importlib.import_module(f"{_PKG}.datasets.{_name}")

for _name in ("base", "llava", "qwen", "minicpm", "internvl"):
    sys.modules[f"{__name__}.models.{_name}"] = importlib.import_module(f"{_PKG}.models.{_name}")

sys.modules[f"{__name__}.experiments.runner"] = importlib.import_module(f"{_PKG}.experiments.runner")

_vlm = importlib.import_module(_PKG)
from vlm_benchmark import DEFAULT_CONFIG, finalize_config  # noqa: E402

__all__ = ["DEFAULT_CONFIG", "finalize_config"]
