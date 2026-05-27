# VLM benchmark framework

Dataset-agnostic experiment layer for modality ablations on math word problems.

Paper wording note: the "text_only" condition is a vision-disabled VLM condition (not a separate text-only model family).

## Canonical sample

Every adapter returns:

```python
BenchmarkSample(
    question=str,
    answer=str,
    image=PIL.Image | None,
    metadata=dict,  # optional id, reasoning, equation, …
)
```

## Adding a dataset

1. Create `gsm8k_experiment/datasets/mydata.py` implementing `load(cfg) -> list[BenchmarkSample]`.
2. Call `register_adapter(MyAdapter())` at module bottom.
3. Import the module in `gsm8k_experiment/datasets/__init__.py`.
4. Add an `AnswerStrategy` in `answer_parsing.py` (or register in `_STRATEGIES`).
5. Run with `--dataset-type mydata`.

## Model adapter layout

Model wrappers now live in `gsm8k_experiment/models/`:

- `llava.py`
- `qwen.py`
- `minicpm.py`
- `internvl.py`

The shared registry and prompt builder live in `gsm8k_experiment/models/__init__.py` and `gsm8k_experiment/models/base.py`.
The main loop is in `gsm8k_experiment/experiments/runner.py`.

## Config keys

| Key | Purpose |
|-----|---------|
| `dataset_type` | Adapter + answer parser (`gsm8k`, `svamp`, …) |
| `metadata_csv` | GSM8K local metadata (optional) |
| `image_root` | Root for relative image paths |
| `hf_dataset_id` | Hugging Face repo when loading from Hub |
| `num_problems` | Subsample size (`None` = all) |

## GSM8K vs SVAMP today

| | GSM8K | SVAMP |
|---|--------|--------|
| Images | `rendered_images/` via metadata CSV | None (text-only until rendered) |
| Answer format | `#### <n>` + integer match | Plain numeric string |
| Paper conditions | text / image / mismatch | text_only ready |
