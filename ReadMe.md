# Effect of Input Modality on Mathematical Reasoning in Vision-Language Models

Presented at the **USF UR2PhD Symposium 2025** · [View Poster](poster/VLM_GSM8K_Poster.pdf)

---

## Start here

**Confused by two folders (`src/` vs `vlm_benchmark/`)?**  
→ **[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)** — which version to use, in one page.

| I want to… | Go to |
|------------|-------|
| Run Phase 1 on Colab (8 models, n=1319) | [`notebooks/Run_All_Models_Free.ipynb`](notebooks/Run_All_Models_Free.ipynb) + [`docs/COLAB.md`](docs/COLAB.md) |
| Understand architecture | [`docs/CANONICAL.md`](docs/CANONICAL.md) |
| Code with AI / avoid duplicate pipelines | [`CLAUDE.md`](CLAUDE.md) |
| Reproduce 2025 symposium pilot only | [`vlm_benchmark/README.md`](vlm_benchmark/README.md) + HF v1 dataset |

**Contributors:** `git pull` before coding. Do not create parallel experiment frameworks.

---

## Project status

| Track | Status | Images | Scale |
|-------|--------|--------|-------|
| **Full study (current)** | Phase 1 in progress | **v2** — Drive → HF upload | n=1319, 8 models |
| **Symposium pilot (legacy)** | Complete — poster | **v1** — [HF dataset](https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm) | n=100, 2 models |

**Do not mix v1 and v2 results in the same table.**

---

## Overview

We study how **input modality** affects math reasoning in VLMs: same model, different inputs (text-only, rendered image, mismatch), measuring accuracy drops and text dominance under conflict.

### Symposium pilot findings (n=100, dataset v1)

- Visual input drops accuracy (−25pp Qwen2-VL-2B, −19pp LLaVA-1.6)
- Mismatch: models follow text in >75% of cases
- See tables below — **pilot only**, not the full-scale study

### Full study (current)

- GSM8K test split (1319), McNemar + bootstrap CIs + Cohen's h
- 8 open VLMs via Colab (`src/` pipeline)
- See Google Drive `vlm_research_results/` for new numbers

---

## How to run (current)

### Colab — Phase 1 (recommended)

1. Open [`notebooks/Run_All_Models_Free.ipynb`](notebooks/Run_All_Models_Free.ipynb) on [Colab](https://colab.research.google.com) from GitHub
2. **Runtime → T4 GPU**
3. Mount Drive → `vlm_research_results/`
4. Set `MODELS_TO_RUN` per session: `[0,1,2]` → `[3,4,5]` → `[6,7]`

Full steps: [`docs/COLAB.md`](docs/COLAB.md)

### CLI

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --config configs/default.yaml --num-problems 10
python scripts/run_multi_benchmark.py --benchmarks gsm8k,svamp --num-problems 50
```

### GSM8K conditions (current Phase 1)

| Condition | Description |
|-----------|-------------|
| Text-only | Vision disabled, text prompt |
| Rendered image | Image only (`src/rendering.py` protocol) |
| Mismatch | Imageᵢ + textᵢ₊₁ |

Aligned text+image exists only in legacy `vlm_benchmark/` (not Phase 1 notebook).

### McNemar (paired conditions)

```bash
python scripts/compare_mcnemar.py results/a.csv --col-a correct_text --csv-b results/b.csv --col-b correct_rendered
```

---

## Datasets

### Phase 1 — GSM8K (current)

- **HF:** [vlm-modality-research/gsm8k-rendered-vlm-v2](https://huggingface.co/datasets/vlm-modality-research/gsm8k-rendered-vlm-v2)
- **Columns:** `problem_id`, `question`, `answer`, `split`, `image`
- **Renderer:** `src/rendering.py` — 900px, raw question, `q000.png` naming

### Phase 3 — Multi-benchmark rendered datasets

| Dataset | HuggingFace | Problems |
|---------|-------------|---------|
| SVAMP | [vlm-modality-research/svamp-rendered-vlm-v1](https://huggingface.co/datasets/vlm-modality-research/svamp-rendered-vlm-v1) | 300 |
| AQuA-RAT | [vlm-modality-research/aqua-rat-rendered-vlm-v1](https://huggingface.co/datasets/vlm-modality-research/aqua-rat-rendered-vlm-v1) | 254 |
| MATH-500 | [vlm-modality-research/math-rendered-vlm-v1](https://huggingface.co/datasets/vlm-modality-research/math-rendered-vlm-v1) | 500 |

All datasets include `problem_id`, `question`, `answer`, `split`, `image`. Use `--hf-images` with `run_benchmark.py` to load canonical images instead of re-rendering locally.

### v1 — symposium pilot (legacy)

**https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm**

| Artifact | Description |
|----------|-------------|
| `rendered_images/` | `q0000.png` … `q1318.png` |
| `data/gsm8k_metadata_clean.csv` | Metadata |
| `data/render_config.json` | 672px, "Solve this step-by-step" prefix |

Regenerate: `python scripts/render_gsm8k.py` · Details: [`docs/DATASET_README.md`](docs/DATASET_README.md)

---

## Repository structure

```
vlm-modality-research/
├── CLAUDE.md                 # AI assistant rules
├── docs/
│   ├── GETTING_STARTED.md    # ← read this if confused
│   ├── CANONICAL.md          # architecture truth
│   ├── COLAB.md
│   ├── FRAMEWORK.md
│   └── DATASET_README.md
├── src/                      # CURRENT — models, eval, benchmarks
├── configs/default.yaml      # CURRENT — 8 models, render settings
├── notebooks/
│   ├── Run_All_Models_Free.ipynb   # CURRENT — Phase 1
│   └── README.md             # which notebook when
├── scripts/
│   ├── run_benchmark.py      # CURRENT — uses src/
│   ├── run_multi_benchmark.py
│   ├── compare_mcnemar.py
│   └── render_gsm8k.py       # LEGACY — v1 images only
├── vlm_benchmark/            # LEGACY — symposium package
├── results/                  # Pilot CSVs (n=100) — do not mix with Drive
└── poster/
```

---

## Pilot results (symposium, n=100, v1)

**Table 1: Zero-Shot Accuracy**

| Model | Text-Only | Rendered Image | Drop |
|-------|-----------|----------------|------|
| Qwen2-VL-2B | 55% | 30% | −25pp |
| LLaVA-1.6-7B | 40% | 21% | −19pp |

**Table 2: Modality Preference Under Conflict**

| Model | Follows Image | Follows Text | Equal |
|-------|--------------|--------------|-------|
| Qwen2-VL-2B | 20 | 79 | 1 |
| LLaVA-1.6-7B | 23 | 76 | 0 |

---

## Evaluation

- GSM8K: extract `#### <answer>` or last number; integer match after rounding
- Errors: `correct`, `arithmetic_error`, `reasoning_error`, `no_number`, `vision_error`
- Full study adds: bootstrap CIs, McNemar, Cohen's h (`src/evaluation.py`)

---

## Authors

**Rodela Ghosh** · University of South Florida  
**Aviral Gupta** · University of South Florida

## Acknowledgements

PALM Lab, UR2PhD program, mentors Ocean Monjur and Shrestha Datta, PI Anshuman Chhabra.
