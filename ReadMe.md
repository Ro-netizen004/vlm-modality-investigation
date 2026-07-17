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
| **3 — Multi-benchmark** | SVAMP, MATH-500, AQuA-RAT (Protocol A) + MathVista, AI2D, ChartQA, ScienceQA (Protocol B) | 7/8 models complete |
| **4 — Noise ablation** | Rendered-image robustness across 10 corruption levels | 4-model contrast (resilient vs vulnerable) |
| **5 — Prompt sensitivity** | Can prompting shift modality preference? | In progress |
| **6 — Legibility (image arm)** | Modality preference under **image** degradation (mismatch × noise): does text preference track legibility, or is it a fixed prior? Behavioral preference **+ conditional-log-likelihood (CLL) arbitration margin** — a ceiling-free graded measure (6 open models) — **+ frontier binary** (GPT-5.6-Luna). Noise applied to the **canonical HF renders** (Level 0 = the main-experiment image). | GSM8K complete; SVAMP CLL + frontier in progress |
| **7 — Mirror arm (text degradation)** | Symmetric counterpart: hold the **image** clean, degrade the **text**, measure the trust shift. With Phase 6 this is a psychophysics-style test of whether VLMs are **reliability-weighted observers** (`src/text_noise.py`) | In progress (degradation module built) |
| **8 — Mechanistic (attention × legibility)** | Mean text→image attention vs. corruption level — a *ceiling-free* complement to Phases 6–7 (Qwen family; extends Hua et al.'s router heads onto the reliability axis) | Planned |

**Target venue:** EACL 2027 (ARR, Aug 3 2026) — Findings the realistic landing,
main track the stretch. Framed as a **psychophysics-style test of reliability-weighted
modality arbitration**: a rational observer down-weights whichever channel is
unreliable (blurry eye → trust the other). We degrade the **image** under conflict
(Phase 6) and, symmetrically, the **text** (Phase 7), asking whether preference tracks
each channel's legibility. **Headline so far:** text-dominance is *largely invariant* to
image legibility — a conditional-log-likelihood analysis (scale-validated, 0.82 agreement
over n=15,257) shows only **1 of 6** open models robustly down-weights the degraded image
in probability space — i.e. a **fixed, redundancy-driven prior, not a reliability-weighted
observer** (with Qwen2.5-VL-7B the notable exception). Differentiated from prior conflict
work that degrades the *text* (Deng et al.) or varies *difficulty* (Pezeshkpour et al.);
Phase 8 adds the mechanistic account. Related work (conflict: Hua et al., Nguyen et al.;
image-degradation reliance: "Diagnosing Visual Ignorance"; robustness: VLM-RobustBench,
Common Corruptions) is cited and differentiated.

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
text-only by **+10–63pp** on every model and benchmark (largest on ChartQA, where
text-only accuracy is near zero).

**Legibility (image arm, CLL ceiling-cracker)** — behavioral preference is pinned
near ceiling and barely moves as the image degrades. The graded **CLL margin** (scale-
validated at 0.82 agreement, n=15,257) shows only **Qwen2.5-VL-7B** robustly down-weights
the degraded image (median margin 0.78→1.54 nats, *p*≈4e-26); the other five open models
are flat — text-dominance is a **fixed prior, not reliability-weighted arbitration**.
Frontier **GPT-5.6-Luna**: 95.9→100% text as the image degrades (binary; API exposes no
logprobs, so no CLL).

---

## Repository layout

```
src/                  # models (VLMModel: 8 open + OpenAI/Gemini API paths), evaluation,
│                     #   image noise (noise.py), TEXT noise for the mirror arm (text_noise.py)
scripts/              # runners, rescore, error analysis, GAIVI/SLURM
│                     #   legibility: run_legibility.py (--score-cll for CLL, --channel for mirror)
│                     #   CLL: validate_cll.py, analyze_cll.py, plot_cll.py
│                     #   frontier: analyze_confidence.py (answer-confidence trajectory, API models)
notebooks/            # Colab/Kaggle runners + analysis notebooks
configs/              # model + rendering configuration
results/
├── phase1/<model>/   # GSM8K results + per-model analysis (8 models)
├── phase2_error_analysis_summary.json   # cross-model disagreement analysis
├── phase3/<model>/   # multi-benchmark results (Protocol A/B, 7 models)
├── phase4/<model>/   # noise ablation results (4-model contrast)
├── phase6_legibility/[<benchmark>/]<model>/  # image-arm preference; per-level CSV/JSON,
│                                             #   level_*.cll.jsonl (CLL margins), rescore/,
│                                             #   *.logprobs.jsonl (API models). gsm8k at root.
└── phase8_attention/[<benchmark>/]<model>/   # text→image attention vs legibility (Qwen family)
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

# Phase 6 — legibility (image arm): behavioral preference + CLL margin
python scripts/run_legibility.py --benchmark gsm8k --models <name> --noise-levels 0 2 4 5
python scripts/validate_cll.py --model <name> --benchmark gsm8k --n 30   # gate: sign-agreement ≥0.75
python scripts/run_legibility.py --benchmark gsm8k --score-cll --models <name> --noise-levels 0 2 4 5
python scripts/analyze_cll.py && python scripts/plot_cll.py              # tables + figure

# frontier (API): set OPENAI_API_KEY / GEMINI_API_KEY; GPT-5.6-Luna, Gemini-2.5-Flash-Lite
python scripts/run_legibility.py --benchmark gsm8k --models GPT-5.6-Luna --noise-levels 0 2 4 5
python scripts/analyze_confidence.py results/phase6_legibility/GPT-5.6-Luna --plot
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
- **Legibility arms:** image degradation (Phase 6, `src/noise.py`) and text
  degradation (Phase 7 mirror, `src/text_noise.py`), each a monotonic 0/2/4/5 ladder
- **Scoring:** numeric match; 5-category mismatch (image / text / neither /
  ambiguous / invalid); reasoning-trace rescore of *neither* trials
- **Graded reliance (ceiling-cracker):** conditional-log-likelihood arbitration margin
  `CLL(text) − CLL(image)` under a direct-answer scaffold (open models only; teacher-
  forced, so unavailable to API models). Validated against behavior at scale
  (sign agreement 0.82, n=15,257). API models: behavioral preference + answer-confidence
  trajectory where the API exposes logprobs (`analyze_confidence.py`)
- **Statistics:** McNemar's test, bootstrap + Clopper-Pearson CIs, Cohen's *h*
  (`src/evaluation.py`); Mann–Whitney + Spearman trend for CLL trajectories
- **Models:** 8 open VLMs (2B–8B; greedy, bfloat16, no quantization) + frontier API
  (GPT-5.6-Luna; Gemini path). CLL covers the 6 open models with a standard `generate`
  interface (not the two custom-chat models, nor the API models)

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
