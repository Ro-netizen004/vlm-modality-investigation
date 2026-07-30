# ChartQA plain-table representation ablation

This ablation changes only the visual representation of the chart-supported
facts. The original chart is replaced by a plain table image rendered from the
official table data recorded in the curation workbook.

The following remain fixed:

- shared question;
- chart-supported answer;
- conflicting evidence-bearing report and report-supported answer;
- Source A/B assignment;
- prompt, degradation arm, legibility level, decoding, and CLL candidates.

The table's answer must **not** be changed to the report answer. Doing so would
make the two sources agree and would no longer test arbitration. Counterfactual
editing remains confined to the report exactly as in ChartQA-Conflict.

Generate the 12-item visual pilot:

```bash
python scripts/prepare_chartqa_table_ablation.py \
  --limit 12 \
  --output-dir outputs/chartqa_table_ablation/pilot12
```

Generate all eligible retained items while recording tables that cannot fit the
matched chart canvas:

```bash
python scripts/prepare_chartqa_table_ablation.py \
  --on-overflow skip \
  --output-dir outputs/chartqa_table_ablation/full229
```

Each output contains rendered PNGs, `manifest.jsonl`, `skipped.jsonl`, and
`summary.json`. Exclusions are determined before model inference.

Validate that the pilot manifest and image checksums can be loaded by the
inference runner:

```bash
python scripts/run_chartqa_conflict.py \
  --prepare-only \
  --report-type evidence \
  --visual-representation plain_table \
  --table-manifest outputs/chartqa_table_ablation/pilot12/manifest.jsonl \
  --num-problems 12
```

Run a small generation smoke test on the clean and heavily degraded endpoints:

```bash
OUT=~/vlm_research_results/phase_control/chartqa_table_smoke12
TABLE_MANIFEST=outputs/chartqa_table_ablation/pilot12/manifest.jsonl

for ARM in image text; do
  python scripts/run_chartqa_conflict.py \
    --models Qwen2.5-VL-7B-Instruct \
    --arm "$ARM" \
    --mode generation \
    --levels 0 5 \
    --num-problems 12 \
    --report-type evidence \
    --visual-representation plain_table \
    --table-manifest "$TABLE_MANIFEST" \
    --output-dir "$OUT"
done
```

For the full experiment, use `full229/manifest.jsonl` with
`--num-problems 229`. Very long two-column source tables are continued in a
second side-by-side panel within the original chart canvas; no facts are
removed. Run the original-chart condition on the same 229 conflict IDs for the
paired representation contrast.

## Verify the spreadsheet and official ChartQA provenance

The frozen curation-workbook inspection is included locally. Audit the complete
spreadsheet-to-image chain with:

```bash
python scripts/audit_chartqa_table_ablation.py --workbook-inspect
```

This matches all 229 entries by `pool_conflict_id`, verifies their questions and
every table cell, checks every PNG checksum, and re-renders every table to test
pixel equality.

For an independent audit against the original source files, extract the
official ChartQA release and run:

```bash
python scripts/audit_chartqa_table_ablation.py \
  --chartqa-root "/path/to/extracted/ChartQA"
```

The root may either contain `ChartQA Dataset/` or be the `ChartQA Dataset`
directory itself. This mode compares every parsed official CSV cell with
`official_table_data`. Both modes write
`outputs/chartqa_table_ablation/audit.json` and exit nonzero on any mismatch.

## Build or publish the paired-image dataset

The publisher refuses to run until a complete 229-item provenance audit passes.
Each published row records whether the audit source was the frozen workbook or
the independent official CSV release:

```bash
python scripts/publish_chartqa_table_ablation_dataset.py --build-only
```

After inspecting the local dataset, publish a new dataset version rather than
modifying the frozen dataset used for the paper's existing results:

```bash
python scripts/publish_chartqa_table_ablation_dataset.py \
  --repo vlm-modality-research/chartqa-evidence-conflict-table-v1 \
  --public
```

Each row contains both `chart_image` and `table_image`, along with the unchanged
question, conflicting report, candidate answers, official table data, source
CSV path, and frozen original-manifest hash.
