"""Backward-compatible runner import shim."""

from .experiments.runner import run_experiment, summarize_results

__all__ = ["run_experiment", "summarize_results"]
