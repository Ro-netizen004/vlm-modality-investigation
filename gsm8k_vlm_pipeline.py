"""
Deprecated monolithic pipeline.

Use the modular package instead:

  python scripts/run_gsm8k_benchmark.py --help

Modules:
  gsm8k_experiment/models/         - model-family adapters + registry
  gsm8k_experiment/eval.py         - scoring utilities
  gsm8k_experiment/experiments/runner.py - experiment loop
  notebooks/analyze_results.ipynb  - plots / analysis only
"""

raise SystemExit(
    "gsm8k_vlm_pipeline.py is deprecated. Run: python scripts/run_gsm8k_benchmark.py --help"
)
