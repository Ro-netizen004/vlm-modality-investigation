# Canonical project reference

**Audience:** humans and AI assistants. If anything disagrees with this file, treat this as wrong and fix the other doc.

Last aligned with: Phase 1 Colab runs (`src/` + `Run_All_Models_Free.ipynb`).

---

## Active pipeline

```
openai/gsm8k (test)
    → rendered images (v2 protocol, Drive or HF)
    → src/models.VLMModel
    → 3 conditions (text_only, rendered_image, mismatch)
    → results/<model>/gsm8k_results.csv + statistics.json
```

**Entry points:**

- Colab: `notebooks/Run_All_Models_Free.ipynb`
- Local/CLI: `python scripts/run_benchmark.py --config configs/default.yaml`
- Multi-benchmark: `python scripts/run_multi_benchmark.py`

**Legibility + mechanistic pipeline (Phases 6–8):**

```
canonical HF renders (Protocol A: gsm8k/svamp/math)
    → src/noise.apply_noise_to_images   (noise ON TOP of canonical; Level 0 == baseline)
    → Phase 6  scripts/run_legibility.py                    → text-preference vs IMAGE level
    →          scripts/run_legibility.py --channel text     → text-preference vs TEXT level (Phase 7 mirror)
    →          scripts/run_legibility.py --score-cll [--channel text]  → CLL arbitration margin (both arms)
    → Phase 8  scripts/run_attention_legibility.py          → text→image attention vs level (Qwen only)
    → scripts/plot_legibility.py [--attention-results-dir ...]
```

- **Phase 6 (image arm):** degrade the image, hold text clean. Results at the
  `output-dir` root. Fan-out: `scripts/gaivi_run_legibility_parallel.sh`.
- **Phase 7 (text arm / mirror):** hold the image clean (level-0 render), degrade the
  **text** via `src/text_noise.degrade_text` (monotonic ladder 0/2/4/5), same mismatch
  trials and scoring. Enabled by `--channel text`; results namespaced under
  `text_legibility/`. Fan-out: `scripts/gaivi_run_text_legibility_parallel.sh` (one job
  per model, all levels together — text degradation is a string op, no per-level image
  cost). Both arms also support the CLL arbitration margin (`--score-cll`), which honors
  `--channel`; the text arm corrupts with `seed=txt_idx` so the CLL run scores the *same*
  string the generation run saw and the reasoning-label join stays coherent.
- Fan-out prep job renders the canonical images once to avoid races (the text arm needs
  only the level-0 clean render).
- **Do not** re-render text for legibility — apply noise to the canonical HF image
  (`apply_noise_to_images`), not `render_noisy_images`, so Level 0 matches Phase 1/3.
- **Phase 8 (attention):** `run_attention_legibility.py` needs `attn_implementation="eager"`;
  reliable only on the Qwen family so far. Run `--smoke` before any full grid.

---

## Legacy pipeline (still in repo, not used for Phase 1)

```
vlm_benchmark/datasets → vlm_benchmark/experiments/runner.py
    → 4 modes including text_and_image
    → HF v1 images (RodelaG/gsm8k-rendered-vlm)
    → CLI: would be run_benchmark.py --dataset-type gsm8k --mode ...
```

The CLI flags above **do not work** on current `scripts/run_benchmark.py` (it was replaced to use `src/`). To revive this path, restore the `vlm_benchmark` CLI explicitly — do not fork again.

---

## Image datasets

### v1 — symposium / pilot

- **Hub:** [RodelaG/gsm8k-rendered-vlm](https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm)
- **Renderer:** `scripts/render_gsm8k.py`
- **Width:** 672px | **Prefix:** `"Solve this step-by-step:\n\n"` | **Names:** `q0000.png`

### v2 — full study (Phase 1)

- **Hub:** [vlm-modality-research/gsm8k-rendered-vlm-v2](https://huggingface.co/datasets/vlm-modality-research/gsm8k-rendered-vlm-v2)
- **Columns:** `problem_id`, `question`, `answer`, `split`, `image`
- **Renderer:** `src/rendering.py` (`render_text_to_image`)
- **Width:** 900px | **Prefix:** none | **Names:** `q000.png` … `q999.png`, `q1000.png` … `q1318.png`

### Phase 3 rendered datasets

| Dataset | Hub | Problems |
|---------|-----|---------|
| SVAMP | [vlm-modality-research/svamp-rendered-vlm-v1](https://huggingface.co/datasets/vlm-modality-research/svamp-rendered-vlm-v1) | 300 |
| AQuA-RAT | [vlm-modality-research/aqua-rat-rendered-vlm-v1](https://huggingface.co/datasets/vlm-modality-research/aqua-rat-rendered-vlm-v1) | 254 |
| MATH-500 | [vlm-modality-research/math-rendered-vlm-v1](https://huggingface.co/datasets/vlm-modality-research/math-rendered-vlm-v1) | 500 |

All Phase 3 datasets include `problem_id`, `question`, `answer`, `split`, `image` columns.

**Do not compare v1 and v2 numbers as the same experiment.**

---

## Results layout (Phase 1)

```
vlm_research_results/          # Google Drive
├── rendered_images/
├── Qwen2-VL-2B-Instruct/
│   ├── gsm8k_results.csv
│   ├── statistics.json
│   ├── statistics_report.txt
│   ├── error_summary.csv
│   └── mismatch_results.csv
├── llava-v1.6-mistral-7b-hf/
│   └── ...
└── cross_model_summary.json   # after all 8 models
```

---

## Git workflow

1. `git pull origin main` before starting work  
2. No large AI-generated dumps without reading existing modules  
3. Merge conflicts on `scripts/run_benchmark.py`: **discuss**, don't auto-keep one side  
4. Update this file + `CLAUDE.md` + `ReadMe.md` when the canonical path changes  

---

## Doc maintenance

- [x] `ReadMe.md`, `FRAMEWORK.md`, `GETTING_STARTED.md` — aligned with current vs legacy  
- [x] HF v2 dataset — [vlm-modality-research/gsm8k-rendered-vlm-v2](https://huggingface.co/datasets/vlm-modality-research/gsm8k-rendered-vlm-v2)  
- [x] HF dataset README — published on Hub  
- [x] Phase 3 datasets — SVAMP, AQuA-RAT, MATH-500 uploaded to HF org  
## Prompt-role control

The canonical role-framing control uses `scripts/run_legibility.py --prompt-role neutral`
for both image- and text-degradation arms. The neutral scaffold names both inputs as
sources and counterbalances their A/B labels. Per-model output directories contain
`experiment_config.json`; the runner refuses to mix a different model, channel, prompt
version, sample size, or dataset fingerprint into an existing directory. Analyze the
matched original-versus-neutral CLL contrast with `scripts/analyze_role_control.py`.
