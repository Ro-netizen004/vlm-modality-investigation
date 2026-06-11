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

- **Hub:** TBD after team uploads Drive folder
- **Renderer:** `src/rendering.py` (`render_text_to_image`)
- **Width:** 900px | **Prefix:** none | **Names:** `q000.png` (3-digit)
- **Source of truth until HF upload:** Google Drive `vlm_research_results/rendered_images/`

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
- [ ] HF v2 dataset URL — fill in after Drive upload  
