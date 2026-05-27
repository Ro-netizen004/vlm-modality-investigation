# Effect of Input Modality on Mathematical Reasoning in Vision-Language Models

Presented at the **USF UR2PhD Symposium 2025** · [View Poster](poster/VLM_GSM8K_Poster.pdf)

---

## Overview

This project investigates how input modality affects mathematical reasoning in vision-language models (VLMs). Using 100 problems from the GSM8K benchmark, we evaluate two models across three controlled conditions — text-only, rendered image, and a novel modality mismatch condition — to isolate the effect of input format on reasoning performance.

**Key findings:**
- Visual input causes substantial accuracy drops in both models (−25pp for Qwen2, −19pp for LLaVA)
- Arithmetic errors increase significantly under image input; vision/OCR errors remain negligible
- In the mismatch condition, both models follow the text modality in over 75% of cases, revealing strong implicit text dominance

---

## Dataset

### GSM8K (benchmark)

Evaluations use the **GSM8K** grade-school math word-problem benchmark. Text and labels are loaded from Hugging Face as [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (config `main`, test split).

### Rendered GSM8K images (Hugging Face)

Pre-rendered GSM8K **test** images (full split), metadata, and rendering config are published here:

**https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm**

| Artifact | Description |
|----------|-------------|
| `rendered_images/` | PNGs named `q0000.png` … `q1318.png` |
| `data/gsm8k_metadata_clean.csv` | Canonical metadata table (`id`, `question`, `answer`, `image`, `reasoning`) |
| `data/render_config.json` | Full rendering protocol and provenance |

Regenerate locally with `scripts/render_gsm8k.py`, or download from the Hub for notebook evaluation.

For release hygiene: keep dataset artifacts on Hugging Face (`RodelaG/gsm8k-rendered-vlm`) and keep this GitHub repo focused on code, notebooks, and documentation.

### Dataset loader (`rendered_gsm8k/dataset.py`)

This file is the **reproducibility API** for the rendered set—not training or rendering code. It:

1. Reads `data/gsm8k_metadata_clean.csv` and checks `id` is 0…N-1 (GSM8K test order)
2. Verifies every PNG in `rendered_images/` exists
3. Returns a Hugging Face `DatasetDict` with a single **`test`** split

```python
from rendered_gsm8k import download_from_hub, load_rendered_gsm8k

data_dir = download_from_hub()   # or a local path after snapshot_download / render
ds = load_rendered_gsm8k(data_dir)
test = ds["test"]
# test[i]["image"], test[i]["question"], test[i]["answer"]
```

---

## Models

| Model | Size | Type |
|-------|------|------|
| Qwen2-VL-2B-Instruct | 2B | Open-source, local |
| LLaVA-v1.6-Mistral-7B | 7B | Open-source, local |

---

## Experimental Conditions

| Condition | Input | Vision Encoder |
|-----------|-------|----------------|
| 1 — Text-Only | Raw problem text | Disabled |
| 2 — Rendered Image | Clean PIL-rendered PNG | Enabled |
| 3 — Modality Mismatch | Image of problem i + text of problem i+1 | Enabled |

---

## Results

**Table 1: Zero-Shot Accuracy**

| Model | Text-Only | Rendered Image | Drop |
|-------|-----------|----------------|------|
| Qwen2-VL-2B | 55% | 30% | −25pp |
| LLaVA-1.6-7B | 40% | 21% | −19pp |

**Table 2: Modality Preference Under Conflict (n=100)**

| Model | Follows Image | Follows Text | Equal |
|-------|--------------|--------------|-------|
| Qwen2-VL-2B | 20 | 79 | 1 |
| LLaVA-1.6-7B | 23 | 76 | 0 |

---

## Repository Structure

```
vlm-modality-research/
├── README.md
├── requirements.txt
├── docs/
│   └── DATASET_README.md
├── data/                          # generated metadata/configs (ignored by git)
│   ├── gsm8k_metadata_clean.csv
│   └── render_config.json
├── scripts/
│   ├── render_gsm8k.py
│   └── fix_paths.py
├── rendered_gsm8k/                # Dataset loader (test split only)
│   └── dataset.py
├── notebooks/
│   ├── VLM_GSM8K_Benchmarking.ipynb
├── rendered_images/               # generated (ignored by git)
├── results/
│   ├── Qwen2-VL-2B-Instruct/
│   │   ├── gsm8k_vlm_results.csv
│   │   ├── error_summary.csv
│   │   └── mismatch_results.csv
│   └── LLaVa_1.6/
│       ├── gsm8k_vlm_results.csv
│       ├── error_summary.csv
│       └── mismatch_results.csv
└── poster/
    └── VLM_GSM8K_Poster.pdf
```

---

## How to Run

### 1) Get rendered GSM8K images

**Option A — Download from Hugging Face (recommended):**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="RodelaG/gsm8k-rendered-vlm",
    repo_type="dataset",
    local_dir="/content/gsm8k-rendered-vlm",
)
# Images: /content/gsm8k-rendered-vlm/rendered_images/q0000.png ...
```

**Option B — Generate locally** with `scripts/render_gsm8k.py` (see below).

The notebooks expect GSM8K problem images to be named deterministically as:

- `q0000.png`, `q0001.png`, … (full GSM8K test set)

This repo includes a renderer script that generates:

- `rendered_images/` (PNG images)
- `data/gsm8k_metadata.csv` (raw metadata from rendering)
- `data/gsm8k_metadata_clean.csv` (canonical metadata for release/evaluation)
- `data/render_config.json` (full rendering protocol + provenance)

Recommended setup (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\scripts\render_gsm8k.py
```

For publication reproducibility, keep `data/render_config.json` together with the released metadata and images.

### 2) Run the evaluation notebooks (Colab)

1. Open the relevant notebook in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Download images from [RodelaG/gsm8k-rendered-vlm](https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm) (see Option A above), or upload your own
4. Set the image directory path in the notebook config (e.g. `IMAGE_DIR = "/content/gsm8k-rendered-vlm/rendered_images"`)
5. Run all cells sequentially

**Dependencies are installed automatically by the first cell.**

---

## Evaluation

- Numeric answer extracted using GSM8K canonical format (`#### <answer>`) with fallback to last number in output
- Answers compared by rounding to nearest integer
- Error classification: `correct`, `arithmetic_error`, `reasoning_error`, `no_number`, `vision_error`
- Mismatch condition scored by absolute numeric distance to each modality's ground truth

---

## Limitations

- 100 problems is a small sample — findings are indicative rather than statistically conclusive
- Both models are relatively small (2B and 7B); behaviour may differ at larger scales
- Rendered images use clean digital text — real-world visual noise not tested
- Evaluation on GSM8K only; generalisability to other math benchmarks not confirmed

---

## Future Work

- Scale to the full GSM8K test set for statistical reliability
- Add significance testing (McNemar's test, chi-squared)
- Evaluate frontier models (GPT-4o, Gemini 2.0) to assess whether text dominance persists at scale
- Implement screenshot condition with controlled noise levels
- Test on additional benchmarks (SVAMP, AQuA-RAT)

---

## References

[1] Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.  
[2] Wang et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191.  
[3] Liu et al. (2024). LLaVA-NeXT: Improved Baselines with Visual Instruction Tuning. arXiv:2310.03744.

---

## Authors

**Rodela Ghosh** · University of South Florida  
**Aviral Gupta** · University of South Florida

---

## Acknowledgements

This project was conducted through the PALM Lab at the University of South Florida under the UR2PhD program. The authors thank graduate mentors Ocean Monjur and Shrestha Datta for their guidance and mentorship. We also thank PI Anshuman Chhabra and the UR2PhD team for providing research training, resources, and the opportunity to present this work.
