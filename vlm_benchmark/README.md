# vlm_benchmark (legacy)

**Status:** Legacy — symposium pilot and dataset adapters.  
**For new experiments use:** `src/` + [`notebooks/Run_All_Models_Free.ipynb`](../notebooks/Run_All_Models_Free.ipynb).

See [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md) and [`docs/CANONICAL.md`](../docs/CANONICAL.md).

## What this package is

Modular experiment layer from the 2025 symposium work:

- Dataset adapters (`datasets/gsm8k.py`, `svamp.py`)
- Model wrappers (`models/`)
- **4 input modes** including `text_and_image` (`experiments/runner.py`)
- HF **v1** images: [RodelaG/gsm8k-rendered-vlm](https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm)

## What it is not

- **Not** used by Phase 1 Colab runs (`src/` is used instead)
- **Not** connected to current `scripts/run_benchmark.py` CLI flags (`--dataset-type`, `--mode`)

## Still useful for

- Reproducing symposium/pilot results (n=100, v1 images)
- `scripts/compare_mcnemar.py` → imports `vlm_benchmark.stats`
- Reference when porting adapters into `src/` later

## Running legacy experiments

Requires restoring or invoking the package API directly (not the current `run_benchmark.py`). Ask the team before investing here — prefer extending `src/` for new work.
