# Rendered GSM8K-VL Dataset

Rendered GSM8K-VL is a multimodal math-reasoning dataset for vision-language model evaluation.  
Each example links:

- a GSM8K word problem (`question`)
- the final numeric answer (`answer`)
- original GSM8K step-by-step reasoning with formatting artifacts removed for consistency (`reasoning`)
- a rendered image path (`image`)

This dataset is intended for controlled experiments comparing text-only and image-based reasoning behavior.

## Canonical Dataset Artifact

The official dataset release uses:

> `data/gsm8k_metadata_clean.csv`

Other files (`gsm8k_metadata.csv`, `gsm8k_metadata_fixed.csv`) are intermediate processing artifacts and are not the canonical release table.

## Source and Provenance

- **Text benchmark source:** [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
- **Source split:** `test`
- **Rendered image naming:** `q0000.png` ... `q1318.png`
- **Image directory:** `rendered_images/`
- **Metadata files (local repo layout):**
  - `data/gsm8k_metadata_clean.csv` (**canonical release table**)
  - `data/gsm8k_metadata.csv` (intermediate/original-style metadata)
  - `data/gsm8k_metadata_fixed.csv` (intermediate path-normalized metadata)

## Recommended Metadata File

Use **`data/gsm8k_metadata_clean.csv`** for local experiments and release.
Treat other metadata CSV files as non-canonical preprocessing artifacts.

### Columns

- `id` (int): index aligned with GSM8K test order
- `question` (string): original GSM8K question text
- `answer` (string/int-like): extracted final answer from `#### ...`
- `image` (string): relative path, e.g. `rendered_images/q0000.png`
- `reasoning` (string): original GSM8K step-by-step reasoning with formatting artifacts removed for consistency (`<<...>>` spans removed)

## Task Definition

Given an image of a rendered GSM8K math word problem, predict the final numeric answer.
The image contains only the question text; no visual reasoning cues beyond text rendering are present.

## Evaluation Protocol

- Primary metric: exact match accuracy on the final numeric answer
- Scoring uses the `answer` field (final answer), not `reasoning`
- `reasoning` is provided for analysis/interpretability, not for official scoring

## Rendering Configuration

Rendering parameters are documented in `data/render_config.json` (local repo).

Typical settings:
- width: 672 px
- font size: 22
- padding: 40
- controlled clean digital rendering (no blur, compression, rotation, or background noise)

## Quick Validation

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/gsm8k_metadata_clean.csv")
missing = [img for img in df["image"] if not Path(img).exists()]
print("Missing images:", len(missing))
print(missing[:10])
```

Expected: `Missing images: 0`

## Minimal Usage (Pandas)

```python
import pandas as pd

df = pd.read_csv("data/gsm8k_metadata_clean.csv")
print(df.columns.tolist())
print(df.iloc[0][["question", "answer", "image"]])
```

## Hugging Face Release

If publishing to a dataset repo (e.g. `RodelaG/gsm8k-rendered-vlm`), include:

- `rendered_images/`
- `data/gsm8k_metadata_clean.csv`
- `data/render_config.json`
- this README as dataset card text

## Citation

If you use this dataset, please cite:

1. GSM8K (Cobbe et al., 2021)
2. This work (GSM8K-VL dataset and associated experiments; Ghosh & Gupta, 2025)
