# Colab guide (Phase 1)

## Quick start

1. [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub**
2. Repo: `https://github.com/Ro-netizen004/vlm-modality-research`
3. Notebook: **`notebooks/Run_All_Models_Free.ipynb`**
4. **Runtime → Change runtime type → T4 GPU**
5. Run all cells; allow Google Drive mount

## Shared Drive folder

Results and images:

```text
My Drive/vlm_research_results/
├── rendered_images/     # ~1319 PNGs (rendered once, then reused)
├── Qwen2-VL-2B-Instruct/
├── llava-v1.6-mistral-7b-hf/
└── ...
```

Team members must use the **same** Drive folder (share access if needed).

## Model batches

| Session | `MODELS_TO_RUN` | Models | ~Time |
|---------|-----------------|--------|-------|
| 1 | `[0, 1, 2]` | Qwen2-VL-2B, LLaVA-1.6-7B, Qwen2.5-VL-7B | ~6.5h |
| 2 | `[3, 4, 5]` | InternVL2-8B, LLaVA-OneVision, Phi-3.5-Vision | ~7.5h |
| 3 | `[6, 7]` | MiniCPM-V-2.6, Idefics3-8B | ~6h |

```python
MODELS_TO_RUN = [3, 4, 5]  # edit per session
NUM_PROBLEMS = None         # None = full 1319
```

Skip models that already have `statistics.json` in their Drive folder.

## After all 8 models

Run the **Cross-Model Comparison** cells at the bottom of the notebook.

## Tips

- Save to Drive before session ends
- If OOM: High-RAM runtime or one model per session
- `git pull` is automatic via notebook clone; for latest docs, re-open notebook from GitHub
- Read [`CLAUDE.md`](../CLAUDE.md) before asking AI to change code

## Not this

- Old `VLM_GSM8K_Benchmarking.ipynb` — deprecated
- HF v1 `gsm8k-rendered-vlm` — symposium pilot only, not Phase 1 images
