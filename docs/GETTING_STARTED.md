# Start here

Confused by two pipelines? Use this page only.

## Which version should I use?

```text
┌─────────────────────────────────────────────────────────────┐
│  What are you doing?                                        │
├─────────────────────────────────────────────────────────────┤
│  Full GSM8K study (8 models, n=1319, Colab)  →  CURRENT   │
│  Multi-benchmark / noise / prompts             →  CURRENT   │
│  Reproduce 2025 symposium pilot (n=100)        →  LEGACY    │
│  4th condition (text + image aligned)          →  LEGACY*   │
└─────────────────────────────────────────────────────────────┘
```

**CURRENT** = `src/` + notebooks (see below)  
**LEGACY** = `vlm_benchmark/` + HF dataset v1 (symposium only)  
\* fourth condition exists only in legacy runner today

---

## Current workflow (Phase 1 — use this)

### Colab (recommended — no local GPU)

1. Open [`notebooks/Run_All_Models_Free.ipynb`](../notebooks/Run_All_Models_Free.ipynb) from GitHub on [Colab](https://colab.research.google.com)
2. **Runtime → T4 GPU**
3. Mount Google Drive → `My Drive/vlm_research_results/`
4. Set `MODELS_TO_RUN` to your batch (e.g. `[3, 4, 5]`)
5. `NUM_PROBLEMS = None` for full 1319
6. Run all cells

Details: [`docs/COLAB.md`](COLAB.md)

### CLI (local or Colab `%shell`)

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --config configs/default.yaml --num-problems 10
python scripts/run_multi_benchmark.py --benchmarks gsm8k,svamp --num-problems 50
```

### Images (v2 — full study)

| Source | When |
|--------|------|
| Drive `vlm_research_results/rendered_images/` | During Colab Phase 1 (team shared folder) |
| Hugging Face v2 | After team upload (URL in `docs/CANONICAL.md`) |

**Do not** use v1 images for Phase 1 numbers. **Do not** mix v1 and v2 in one table.

### GSM8K conditions (current)

1. **Text-only** — vision disabled  
2. **Rendered image** — image only  
3. **Mismatch** — imageᵢ + textᵢ₊₁  

### Results location

Google Drive: `vlm_research_results/<model-name>/gsm8k_results.csv` + `statistics.json`

---

## Legacy workflow (symposium pilot only)

For reproducing the **2025 poster** (n=100, different images):

| Piece | Path |
|-------|------|
| Images | [RodelaG/gsm8k-rendered-vlm](https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm) (v1) |
| Renderer | `scripts/render_gsm8k.py` |
| Package | `vlm_benchmark/` — see [`vlm_benchmark/README.md`](../vlm_benchmark/README.md) |
| Old notebooks | `notebooks/deprecated/` — **deprecated** |

The old CLI (`--dataset-type gsm8k --mode text_and_image`) is **not** wired to `scripts/run_benchmark.py` on current `main`. Use legacy package directly or ask before reviving.

---

## Repo map (simplified)

| Path | Status |
|------|--------|
| `src/` | **Current** — models, eval, rendering, benchmarks |
| `configs/default.yaml` | **Current** — model list, render settings |
| `notebooks/Run_All_Models_Free.ipynb` | **Current** — Phase 1 |
| `notebooks/Error_Analysis.ipynb` | **Current** — CPU analysis |
| `notebooks/Multi_Benchmark_Eval.ipynb` | **Current** — Phase 3 |
| `vlm_benchmark/` | **Legacy** — symposium / adapters |
| `scripts/render_gsm8k.py` | **Legacy** — v1 image protocol |
| `results/` in git | **Pilot** — n=100, do not mix with Drive results |

---

## Before you code

1. `git pull origin main`
2. Read [`CLAUDE.md`](../CLAUDE.md) (AI assistants) and [`docs/CANONICAL.md`](CANONICAL.md) (full reference)
