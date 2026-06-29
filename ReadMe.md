# Text or Image? Task-Conditional Modality Dominance in Vision-Language Models

We study **which modality vision-language models (VLMs) rely on** when their text
and image inputs conflict — and find the answer depends on the task.

> **Key finding.** Modality dominance is *task-conditional*. When an image merely
> re-renders the text (vision is redundant), models follow the **text** 87–100% of
> the time. When the image carries information the text lacks — charts, diagrams —
> models correctly rely on the **image** (+10–54pp). VLMs are not text-first by
> design; they follow whichever modality is most informative for the task.

A core methodological contribution is the **mismatch condition**: we pair the
image of problem *i* with the text of problem *i+1*, directly measuring modality
preference under conflict rather than inferring it from accuracy differences.

---

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| **1 — GSM8K** | 8 VLMs, 3 conditions (text / image / mismatch) | Complete |
| **2 — Error analysis** | Disagreement + difficulty correlates, 8 models | Complete |
| **3 — Multi-benchmark** | SVAMP, MATH-500, AQuA-RAT (Protocol A) + MathVista, AI2D, ChartQA, ScienceQA (Protocol B) | In progress |
| **4 — Noise ablation** | Rendered-image robustness across 10 corruption levels | In progress |
| **5 — Prompt sensitivity** | Can prompting shift modality preference? | Planned |
| **6 — Mechanistic** | Attention analysis of text vs image tokens | Planned |

**Target venue:** EACL 2027 (ARR, Aug 3 2026).

---

## Headline results (GSM8K, N=1319)

**Text-only vs rendered-image accuracy** — two clear groups emerge:

| Group | Models | Accuracy change |
|-------|--------|-----------------|
| Resilient | Qwen2.5-VL-7B, InternVL2-8B, Qwen2-VL-2B | ≤ 3pp (not significant) |
| Vulnerable | Idefics3-8B, LLaVA-1.6, LLaVA-OneVision, MiniCPM, Phi-3.5 | 5–49pp drop (p<0.001) |

**Mismatch condition** — text preference (decidable trials, after reasoning-trace
rescore): **87–100%** across all eight models. Phi-3.5 is the notable exception
that engages the image more often (87%); the rest are 96–100%.

**Protocol B (natural visual benchmarks)** — the pattern reverses: image beats
text-only by **+10–54pp** on every model and benchmark (largest on ChartQA).

---

## Repository layout

```
src/                  # models (VLMModel), evaluation, multi-benchmark engine, noise
scripts/              # benchmark runners, rescore, error analysis, GAIVI/SLURM
notebooks/            # Colab/Kaggle runners + analysis notebooks
configs/              # model + rendering configuration
results/
├── phase1/<model>/   # GSM8K results + per-model analysis (8 models)
├── phase3/<model>/   # multi-benchmark results (Protocol A/B)
└── phase4/<model>/   # noise ablation results
docs/                 # CANONICAL.md (architecture), dataset specs, onboarding
vlm_benchmark/        # legacy symposium-pilot package (kept for reproducibility)
```

> New contributors: read [`docs/CANONICAL.md`](docs/CANONICAL.md) and
> [`CLAUDE.md`](CLAUDE.md) first; `git pull` before coding; do not create parallel
> experiment frameworks.

---

## How to run

**CLI (local / cluster):**
```bash
pip install -r requirements.txt

# GSM8K, 3 conditions, canonical HF images
python scripts/run_benchmark.py --config configs/default.yaml --hf-images

# multi-benchmark (Protocol A/B)
python scripts/run_multi_benchmark.py --benchmarks gsm8k,svamp

# post-processing: reasoning-trace rescore + error analysis
python scripts/rescore_mismatch_reasoning.py --model <name>
python scripts/run_error_analysis.py --models <name>
```

**Cluster (SLURM / GAIVI):** see `scripts/gaivi_*.sh`.
**Colab:** [`notebooks/Run_All_Models_Free.ipynb`](notebooks/Run_All_Models_Free.ipynb) (see [`docs/COLAB.md`](docs/COLAB.md)).

---

## Datasets (HuggingFace)

All rendered datasets are public under
[`vlm-modality-research`](https://huggingface.co/vlm-modality-research):

| Dataset | Problems |
|---------|----------|
| [`gsm8k-rendered-vlm-v2`](https://huggingface.co/datasets/vlm-modality-research/gsm8k-rendered-vlm-v2) | 1,319 |
| [`svamp-rendered-vlm-v1`](https://huggingface.co/datasets/vlm-modality-research/svamp-rendered-vlm-v1) | 300 |
| [`aqua-rat-rendered-vlm-v1`](https://huggingface.co/datasets/vlm-modality-research/aqua-rat-rendered-vlm-v1) | 254 |
| [`math-rendered-vlm-v1`](https://huggingface.co/datasets/vlm-modality-research/math-rendered-vlm-v1) | 500 |

Each includes `problem_id`, `question`, `answer`, `split`, `image`. Noise-corruption
images (Phase 4) are regenerated deterministically from a fixed seed (`src/noise.py`),
not stored.

---

## Methods

- **Conditions:** text-only, rendered-image, mismatch (image_i + text_{i+1})
- **Scoring:** numeric match; 5-category mismatch (image / text / neither /
  ambiguous / invalid); reasoning-trace rescore of *neither* trials
- **Statistics:** McNemar's test, bootstrap + Clopper-Pearson CIs, Cohen's *h*
  (`src/evaluation.py`)
- **Models:** 8 open VLMs (2B–8B), greedy decoding, bfloat16, no quantization

---

## Authors

**Rodela Ghosh** · University of South Florida
**Aviral Gupta** · University of South Florida

## Acknowledgements

This project began as a USF UR2PhD symposium pilot in PALM Lab, and we thank
Anshuman Chhabra, Ocean Monjur, and Shrestha Datta for their mentorship during
that phase. The current full-scale study is carried out in the Computing
Intelligence and Security Lab (CISL) at USF; we thank Prof. Guangjing Wang for
providing GPU resources on the GAIVI cluster.
