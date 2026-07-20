# Instructions for AI assistants (Claude, etc.)

Read this file **before** writing or refactoring code in this repo.

## First steps (every session)

1. `git pull origin main`
2. **Read `docs/PROJECT_STATUS.md`** — live handoff: experiment status, GAIVI commands/paths,
   known issues (MiniCPM version break, `/data` quota, login-node numpy, sbatch gotchas)
3. Skim `docs/CANONICAL.md` (architecture source of truth)
4. Do **not** create parallel pipelines — extend what exists

## What this project is

VLM modality research: same vision-language model under different input conditions (text-only, rendered image, mismatch) on GSM8K and other benchmarks. Goal: reproducible experiments for a workshop/paper.

## Canonical stack (use this — do not reinvent)

| Layer | Location | Notes |
|-------|----------|-------|
| **GSM8K full runs (8 models)** | `notebooks/Run_All_Models_Free.ipynb` | Colab Pro / Kaggle; saves to Drive `vlm_research_results/` |
| **GSM8K CLI** | `scripts/run_benchmark.py` + `configs/default.yaml` | Uses `src/` — **not** `vlm_benchmark/` |
| **Multi-benchmark** | `scripts/run_multi_benchmark.py` + `notebooks/Multi_Benchmark_Eval.ipynb` | Protocol A/B in `src/benchmark_eval.py` |
| **Models & inference** | `src/models.py` (`VLMModel`) | 8 open models; greedy decode (`do_sample=False`) |
| **Stats** | `src/evaluation.py` | McNemar, bootstrap CI, Cohen's h |
| **McNemar CLI** | `scripts/compare_mcnemar.py` | Uses `vlm_benchmark.stats` — OK to keep |
| **Legacy package** | `vlm_benchmark/` | Dataset adapters, 4-mode runner — **do not duplicate**; integrate or deprecate explicitly |

## Images (critical — do not get this wrong)

Two datasets exist; **never mix them in one results table**:

| Dataset | Hub ID | Protocol |
|---------|--------|----------|
| **v1 (symposium pilot)** | `RodelaG/gsm8k-rendered-vlm` | 672px, `"Solve this step-by-step"` prefix, `q0000.png` |
| **v2 (full study)** | [vlm-modality-research/gsm8k-rendered-vlm-v2](https://huggingface.co/datasets/vlm-modality-research/gsm8k-rendered-vlm-v2) | 900px, raw question, `src/rendering.py` |

**Rules:**

- Phase 1 Colab results use **v2** images from Drive `rendered_images/` (or HF v2 once uploaded).
- Do **not** call `render_all_images()` if canonical PNGs already exist — use `load_image()` only.
- Do **not** change render settings without updating `data/render_config.json` and re-uploading HF.

## GSM8K conditions (current Phase 1)

Three conditions per run (not four):

1. `text_only` — vision disabled, text prompt  
2. `rendered_image` — image only  
3. `mismatch` — image_i + text_{i+1}  

Aligned `text_and_image` exists only in `vlm_benchmark/experiments/runner.py` — not in `Run_All_Models_Free.ipynb`. Do not add a fourth condition silently; discuss with team first.

## Hard rules (prevent duplicate-framework bugs)

- **Never** replace `scripts/run_benchmark.py` with an unrelated implementation — extend `src/` or restore `vlm_benchmark` via explicit team decision only.
- **Never** add a second package root (`src2/`, `benchmark/`, etc.) — extend `src/` or `vlm_benchmark/`.
- **Never** resolve git merge conflicts by keeping "local comprehensive version" without reading remote `vlm_benchmark/` changes.
- **Never** hardcode `load_dataset('gsm8k')` — use `openai/gsm8k` config `main`.
- After substantive changes: update `docs/CANONICAL.md` and `ReadMe.md` in the **same PR**.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `Run_All_Models_Free.ipynb` | **Primary** — Phase 1, 8 models × GSM8K |
| `Error_Analysis.ipynb` | CPU; reads saved CSVs from Drive |
| `Multi_Benchmark_Eval.ipynb` | Phase 3 benchmarks |
| `Noise_Ablation.ipynb` | Phase 4 |
| `Prompt_Sensitivity.ipynb` | Phase 5 |
| `Mechanistic_Analysis.ipynb` | Attention/hidden-state probes (feeds Phase 7) |
| `notebooks/deprecated/*.ipynb` | **Deprecated** — do not extend |

**Phase map (canonical numbering — keep consistent with `ReadMe.md`):**
1 GSM8K · 2 Error analysis · 3 Multi-benchmark · 4 Noise ablation ·
5 Prompt sensitivity · 6 Legibility image arm (`scripts/run_legibility.py`; `--score-cll` = CLL margin) ·
7 Mirror arm — text degradation (`src/text_noise.py` + `run_legibility.py --channel text`; built, runs pending) ·
8 Mechanistic attention × legibility (`scripts/run_attention_legibility.py`, Qwen family).

## Reproducibility checklist (new runs)

- Pin `transformers` / `torch` in `requirements.txt` when changing env  
- Record git commit + HF dataset revision in results `statistics.json`  
- `NUM_PROBLEMS = None` for full 1319; label pilots explicitly  
- Save results to Drive before Colab session ends  

## Human docs (update when architecture changes)

- `docs/CANONICAL.md` — architecture & dataset truth  
- `docs/DATASET_README.md` — HF dataset specs (v1 vs v2)  
- `ReadMe.md` — public-facing quick start  
