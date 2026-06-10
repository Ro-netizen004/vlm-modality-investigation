"""
Deprecated monolithic pipeline.

Use the modular package instead:

  python scripts/run_benchmark.py --help

Modules:
  vlm_benchmark/models/            - model-family adapters + registry
  vlm_benchmark/eval.py            - scoring utilities
  vlm_benchmark/experiments/runner.py - experiment loop
  notebooks/analyze_results.ipynb  - plots / analysis only
"""

raise SystemExit(
    "gsm8k_vlm_pipeline.py is deprecated. Run: python scripts/run_benchmark.py --help"
)
