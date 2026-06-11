# Notebooks

## Use these (current)

| Notebook | GPU? | Purpose |
|----------|------|---------|
| **`Run_All_Models_Free.ipynb`** | Yes | **Phase 1** — 8 models × GSM8K (n=1319). Primary entry point. |
| `Error_Analysis.ipynb` | No | Disagreement / difficulty analysis on saved CSVs |
| `Multi_Benchmark_Eval.ipynb` | Yes | Phase 3 — SVAMP, MathVista, etc. |
| `Noise_Ablation.ipynb` | Yes | Phase 4 — image degradation curve |
| `Prompt_Sensitivity.ipynb` | Yes | Phase 5 — prompt strategies |
| `Mechanistic_Analysis.ipynb` | Yes | Phase 6 — attention / OCR (appendix) |
| `Full_GSM8K_Benchmark.ipynb` | Yes | Single-model GSM8K via `src/` (alternative to Run_All) |
| `Kaggle_Benchmark.ipynb` | Yes | Same as above on Kaggle GPU |

Open from GitHub: `https://github.com/Ro-netizen004/vlm-modality-research`

**Drive output:** `My Drive/vlm_research_results/`  
**Config:** `MODELS_TO_RUN = [0,1,2]` then `[3,4,5]` then `[6,7]`

## Deprecated (do not use for new work)

| Notebook | Why |
|----------|-----|
| `VLM_GSM8K_Benchmarking.ipynb` | Monolithic pre-`src/` workflow |
| `LLaVA16_GSM8K_Benchmarking.ipynb` | Same |
| `analyze_results.ipynb` | Expects old `vlm_benchmark` CSV layout |

See [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md).
