# src (current)

**Status:** Canonical code for Phase 1+ experiments.

## Modules

| Module | Role |
|--------|------|
| `models.py` | `VLMModel` — 8 open VLMs, 4-bit quant |
| `evaluation.py` | Accuracy, errors, McNemar, bootstrap CI, Cohen's h |
| `rendering.py` | v2 image render/load (`q000.png` naming) |
| `benchmarks.py` | Load 8 benchmarks |
| `benchmark_eval.py` | Protocol A (text math) / B (visual) |
| `noise.py` | Noise ablation transforms |
| `prompts.py` | Prompt sensitivity strategies |
| `error_analysis.py` | Disagreement correlates |
| `mechanistic.py` | Attention / OCR probes |
| `visualization.py` | Plots |

## Entry points

```bash
python scripts/run_benchmark.py --config configs/default.yaml
python scripts/run_multi_benchmark.py
```

Colab: [`notebooks/Run_All_Models_Free.ipynb`](../notebooks/Run_All_Models_Free.ipynb)

Config: [`configs/default.yaml`](../configs/default.yaml)
