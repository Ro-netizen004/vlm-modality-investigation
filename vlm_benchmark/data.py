"""Backward-compatible dataset loading (prefer vlm_benchmark.datasets.load_dataset)."""

from __future__ import annotations

from .datasets import load_dataset
from .types import BenchmarkSample


def load_benchmark_dataset(cfg: dict) -> tuple[list[str], list[str], list]:
    """Return (questions, references, images) for legacy callers."""
    samples = load_dataset(cfg)
    return (
        [s.question for s in samples],
        [s.answer for s in samples],
        [s.image for s in samples],
    )


def samples_to_parallel(samples: list[BenchmarkSample]) -> tuple[list[str], list[str], list]:
    return (
        [s.question for s in samples],
        [s.answer for s in samples],
        [s.image for s in samples],
    )
