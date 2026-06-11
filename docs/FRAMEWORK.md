# Framework overview

> **New here?** Read [`GETTING_STARTED.md`](GETTING_STARTED.md) first.  
> **Full reference:** [`CANONICAL.md`](CANONICAL.md)

This repo has **two** experiment stacks. Only one is current.

| | **Current (`src/`)** | **Legacy (`vlm_benchmark/`)** |
|--|----------------------|-------------------------------|
| **Use for** | Phase 1–6, full paper | Symposium pilot reproduction |
| **Entry** | `Run_All_Models_Free.ipynb`, `scripts/run_benchmark.py` | Package API only (CLI flags removed) |
| **Images** | v2 (900px, Drive / HF v2) | v1 ([gsm8k-rendered-vlm](https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm)) |
| **GSM8K conditions** | 3 (text, image, mismatch) | 4 (+ aligned text+image) |
| **Docs** | [`src/README.md`](../src/README.md) | [`vlm_benchmark/README.md`](../vlm_benchmark/README.md) |

**Do not add a third stack.** Extend `src/` or explicitly revive legacy with team agreement.

---

## Current stack (`src/`)

```
configs/default.yaml
    → scripts/run_benchmark.py  OR  notebooks/Run_All_Models_Free.ipynb
    → src/models.VLMModel
    → src/evaluation.py (McNemar, bootstrap CI, Cohen's h)
    → results on Google Drive or local results/
```

Multi-benchmark: `scripts/run_multi_benchmark.py` → `src/benchmark_eval.py` (Protocol A/B).

Extensions: `src/noise.py`, `src/prompts.py`, `src/error_analysis.py`, `src/mechanistic.py`.

---

## Legacy stack (`vlm_benchmark/`)

Dataset-agnostic adapters and a 4-mode runner used for the **2025 symposium** (n=100, v1 images).

Paper wording: `text_only` is vision-disabled VLM inference, not a separate text-only model family.

### Legacy sample type

```python
BenchmarkSample(question=str, answer=str, image=PIL.Image | None, metadata=dict)
```

### Legacy McNemar CLI (still works)

```bash
python scripts/compare_mcnemar.py results/a.csv --col-a correct --csv-b results/b.csv --col-b correct
```

Uses `vlm_benchmark/stats.py`. Phase 1 wide CSVs may use `correct_text` / `correct_rendered` column names instead.

---

## Adding a new dataset

**Current path:** register in `src/benchmarks.py`, evaluate via `src/benchmark_eval.py`.

**Legacy path:** `vlm_benchmark/datasets/` + `answer_parsing.py` — only if maintaining pilot stack.

---

## AI-assisted development

Read [`CLAUDE.md`](../CLAUDE.md) before codegen. Update this file when architecture changes.
