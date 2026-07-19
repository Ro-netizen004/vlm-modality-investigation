---
license: mit
task_categories:
  - visual-question-answering
  - question-answering
language:
  - en
tags:
  - vision-language-models
  - multimodal
  - modality-arbitration
  - robustness
  - math-reasoning
  - gsm8k
  - svamp
pretty_name: Modality-Conflict Arbitration Benchmark (v2)
size_categories:
  - 10K<n<100K
---

# Modality-Conflict Arbitration Benchmark (v2)

A controlled benchmark for studying **how a vision-language model arbitrates between
its two input channels when they disagree** — and whether that choice tracks the
*reliability* of each channel.

Each row is a single **conflict** trial: an image of one math problem paired with the
text of a *different* problem. Because the two ground-truth answers are carried side by
side, the model's output alone tells you **which modality it followed** — no reference to
the original pipeline needed.

v2 adds the **text-degradation mirror arm** to the image-degradation arm of v1, so both
sides of the reliability question are testable from one dataset.

---

## The two arms

| `channel` | Image | Text | Conflict introduced via | Question: as the degraded channel worsens, does preference move to the *other* one? |
|-----------|-------|------|-------------------------|----------------------|
| `image` | rendered problem *i*, **degraded** at `degradation_level` | clean problem *(i+1)* | the **image** | rational observer → shifts toward **text** |
| `text`  | rendered problem *i*, **clean** | problem *(i+1)*, **degraded** at `degradation_level` | the **text** (in `prompt`) | rational observer → shifts toward **image** |

A model with a *fixed modality prior* shows a flat preference curve across
`degradation_level`; a **reliability-weighted** model down-weights whichever channel is
degraded. Running both arms separates those two explanations.

---

## Schema

One row = one conflict instance. **Convention:** the `*_question` columns are always the
**clean** problem text (labels); the degradation always lives in the *presented* modality —
the `image` bytes (image arm) or the `prompt` string (text arm). So **`prompt` always
contains the exact, possibly-corrupted text the model was shown.**

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | e.g. `gsm8k-text-L5-0042` (`{source}-{channel}-L{level}-{i}`) |
| `source` | string | `gsm8k` or `svamp` |
| `conflict_type` | string | `mismatch` |
| `channel` | string | `image` or `text` — which channel is degraded |
| `degradation_level` | int32 | `0` (clean) / `2` / `4` / `5` (heaviest) |
| `degradation_name` | string | image arm: `clean`/`blur_light`/`blur_noise`/`heavy_degradation`; text arm: `clean`/`light_corruption`/`medium_corruption`/`heavy_corruption` |
| `image` | Image | the rendered PNG (degraded in the image arm; clean in the text arm) |
| `image_question` | string | clean text of the **image** problem *i* |
| `text_question` | string | clean text of the **conflicting** problem *(i+1)* |
| `image_answer` | string | ground-truth answer for the image problem |
| `text_answer` | string | ground-truth answer for the text problem |
| `prompt` | string | the exact prompt shown to the model (contains the corrupted text in the text arm) |
| `image_problem_id` | int32 | *i* |
| `text_problem_id` | int32 | *(i+1) mod n* |
| `image_seed` | int32 | `42 + i`, the canonical noise seed (image arm) |

Degradation is fully reproducible: the text arm's corrupted `prompt` is
`degrade_text(text_question, degradation_level, seed=text_problem_id)`; the image arm's
pixels are `apply_noise_level(clean_render, degradation_level, seed=image_seed)`.

---

## Quickstart — behavioral scoring (works on any VLM, incl. black-box APIs)

The two answers make "which modality did you follow?" decidable from output text alone.

```python
from datasets import load_dataset

ds = load_dataset("vlm-modality-research/modality-conflict-arbitration-v2", split="train")

def score(row, model_output):
    got_img = row["image_answer"] in model_output   # use your own numeric-match here
    got_txt = row["text_answer"]  in model_output
    if got_txt and not got_img: return "text"
    if got_img and not got_txt: return "image"
    return "neither"

# Run your model on row["prompt"] + row["image"], then:
#   text_preference(level) = P(follow == "text" | decidable) within each channel × level
#   plot it against degradation_level to get the arbitration curve for each arm.
```

> **Tip:** compare like with like — analyse the `image` and `text` channels *separately*,
> each as its own preference-vs-`degradation_level` curve.

### Optional: internal-preference (CLL) scoring
If you have the model **weights** (not just an API), you can also measure the
conditional-log-likelihood *arbitration margin* — teacher-force both candidate answers
under each `prompt` and take `logP(text_answer) − logP(image_answer)`. This gives a
continuous, ceiling-free signal with no `neither` dropout, but requires forward-pass
access, so it is **not** available for closed API models.

---

## Sources & size

| source | problems | levels | channels | rows |
|--------|----------|--------|----------|------|
| gsm8k  | 1319 | 0/2/4/5 | image + text | 10,552 |
| svamp  | 300  | 0/2/4/5 | image + text | 2,400 |
| **total** | | | | **12,952** |

Stimuli are derived from **GSM8K** (Cobbe et al., 2021) and **SVAMP** (Patel et al., 2021);
the images are rendered from the canonical v2 renders
(`vlm-modality-research/gsm8k-rendered-vlm-v2`, `.../svamp-rendered-vlm-v1`). Level 0 is
pixel-identical to those baselines. Text pairing and both answers come from the same
model-agnostic stimuli used and scored in the source experiments.

---

## What you can / cannot test

**Can:** modality bias under conflict; reliability-weighted arbitration in *either* channel;
cross-model / cross-architecture comparison on identical stimuli; OCR/vision robustness vs.
degradation; prompt-intervention studies on fixed stimuli.

**Cannot (by construction):** conflict *cost* vs. an agreement baseline (this is
mismatch-only); fine-grained multimodal *integration* (the two problems are unrelated, so
`neither` is common by design); CLL margins for closed API models (needs weights).

---

## Reproducing the build

```bash
python scripts/build_conflict_dataset.py --out data/conflict_dataset_v2   # both arms
```

Requires the canonical render repos (network) and the model-agnostic stimulus CSVs. See
`scripts/build_conflict_dataset.py` for the exact construction.

---

## Citation

If you use this benchmark, please cite this work (paper in preparation) and the source
datasets it derives from (GSM8K, SVAMP). License: MIT, following the source datasets.
