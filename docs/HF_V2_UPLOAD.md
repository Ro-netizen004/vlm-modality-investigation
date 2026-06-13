# Upload GSM8K rendered images v2 to Hugging Face

**Live dataset:** [vlm-modality-research/gsm8k-rendered-vlm-v2](https://huggingface.co/datasets/vlm-modality-research/gsm8k-rendered-vlm-v2)

Use this guide to **rebuild or refresh** the Hub release from Drive images.  
Do **not** mix with v1 ([`RodelaG/gsm8k-rendered-vlm`](https://huggingface.co/datasets/RodelaG/gsm8k-rendered-vlm)).

---

## Step 1 — Download images from Google Drive

1. Open Drive → `vlm_research_results/rendered_images/`
2. Download the folder to your PC (e.g. `D:\Downloads\rendered_images`)
3. Confirm **1319** PNG files (`q000.png` … or `q0000.png` — script auto-detects)

---

## Step 2 — Build the upload bundle

From the repo root (with venv + `pip install -r requirements.txt`):

```powershell
cd "C:\Users\Tech moon\Documents\GitHub\vlm-modality-research"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/prepare_hf_v2_release.py `
  --images-dir "D:\Downloads\rendered_images" `
  --output-dir ".\hf_gsm8k_v2_upload" `
  --copy-images
```

This creates:

```text
hf_gsm8k_v2_upload/
├── rendered_images/          # PNGs
└── data/
    ├── gsm8k_metadata_clean.csv
    └── render_config.json
```

---

## Step 3 — Create the HF dataset repo

1. Log in: [huggingface.co](https://huggingface.co) → Settings → Access Tokens (write)
2. Create dataset: [huggingface.co/new-dataset](https://huggingface.co/new-dataset)
3. Suggested id: **`vlm-modality-research/gsm8k-rendered-vlm-v2`** (or your org/name)
4. License: same as v1 (e.g. MIT / CC-BY — match your v1 choice)

---

## Step 4 — Upload

### Option A — Web UI

Upload the contents of `hf_gsm8k_v2_upload/` (folders `rendered_images/` and `data/`).

### Option B — CLI (recommended for 1319 images)

```powershell
pip install huggingface_hub

huggingface-cli login

cd hf_gsm8k_v2_upload
huggingface-cli upload vlm-modality-research/gsm8k-rendered-vlm-v2 . --repo-type dataset
```

Replace `vlm-modality-research/gsm8k-rendered-vlm-v2` with your repo id.

---

## Step 5 — Dataset card (README on HF)

On the dataset page, add a short README:

```markdown
---
license: mit
task_categories:
- visual-question-answering
language:
- en
size_categories:
- 1K<n<10K
---

# GSM8K Rendered-VL v2

Rendered GSM8K **test** split (1319) for VLM modality experiments (Phase 1).

**Not interchangeable with v1** (`RodelaG/gsm8k-rendered-vlm`).

| | v1 | v2 (this repo) |
|--|----|----------------|
| Width | 672px | 900px |
| Text on image | "Solve this step-by-step" + question | question only |
| Names | q0000.png | q000.png |

See `data/render_config.json` for full protocol.

Source text: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) test split.
```

---

## Step 6 — Update the code repo

After upload, edit `docs/CANONICAL.md` and `CLAUDE.md`:

```text
v2 Hub: https://huggingface.co/datasets/vlm-modality-research/gsm8k-rendered-vlm-v2
```

Tell Aviral: Phase 1 numbers use **this** revision; v1 is symposium pilot only.

---

## Download in Colab (after upload)

```python
from huggingface_hub import snapshot_download

DATA = snapshot_download(
    "vlm-modality-research/gsm8k-rendered-vlm-v2",
    repo_type="dataset",
    local_dir="/content/gsm8k-rendered-vlm-v2",
)
IMAGE_DIR = f"{DATA}/rendered_images"
```

Optional later: change notebooks to skip `render_all_images()` and load from `IMAGE_DIR` only.
