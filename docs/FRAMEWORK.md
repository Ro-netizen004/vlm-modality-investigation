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

1. Create `vlm_benchmark/datasets/mydata.py` implementing `load(cfg) -> list[BenchmarkSample]`.
2. Call `register_adapter(MyAdapter())` at module bottom.
3. Import the module in `vlm_benchmark/datasets/__init__.py`.
4. Add an `AnswerStrategy` in `answer_parsing.py` (or register in `_STRATEGIES`).
5. Run with `--dataset-type mydata`:

```bash
python scripts/run_benchmark.py --dataset-type mydata --mode text_only
```

(`scripts/run_gsm8k_benchmark.py` is a deprecated CLI alias. `import gsm8k_experiment` is a deprecated package alias for `vlm_benchmark`.)

## Model adapter layout

Model wrappers now live in `vlm_benchmark/models/`:

- `llava.py`
- `qwen.py`
- `minicpm.py`
- `internvl.py`

The shared registry and prompt builder live in `vlm_benchmark/models/__init__.py` and `vlm_benchmark/models/base.py`.
The main loop is in `vlm_benchmark/experiments/runner.py`.

## Paired significance (McNemar)

After two runs on the **same** `problem_id` set (e.g. vision-disabled vs image-only):

```bash
python scripts/compare_mcnemar.py results/run_a_results.csv --col-a correct --csv-b results/run_b_results.csv --col-b correct
```

Legacy wide CSV (one file, two conditions):

```bash
python scripts/compare_mcnemar.py results/LLaVA-1.6/gsm8k_llava16_results.csv --col-a correct_text --col-b correct_rendered --label-a text --label-b image
```

Implementation: `vlm_benchmark/stats.py`.

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
