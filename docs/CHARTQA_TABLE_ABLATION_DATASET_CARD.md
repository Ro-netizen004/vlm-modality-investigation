---
language:
- en
license: other
task_categories:
- visual-question-answering
---

# ChartQA-Conflict chart/table representation ablation

This derivative contains 229 audited ChartQA-Conflict items. Each row provides
two visual renderings of the same official ChartQA facts:

- `chart_image`: the original ChartQA chart;
- `table_image`: a plain table image rendered from the corresponding official
  ChartQA CSV.

The shared question, chart-supported answer, evidence-bearing conflicting
report, report-supported answer, and Source A/B assignment are identical across
the two representations. The table values are not edited toward the conflicting
report. This permits a controlled comparison of genuine-chart and
picture-of-table evidence while holding the underlying facts fixed.

Before publication, `scripts/audit_chartqa_table_ablation.py` records the
provenance scope and checks every row:

1. the recorded cells equal the frozen curation workbook or, when supplied,
   the referenced official ChartQA CSV;
2. the saved table-image checksum matches the manifest;
3. re-rendering those cells reproduces the saved table-image pixels.

One ChartQA-Conflict item (`conflict_id=45`) is excluded because the final report
audit found that its report did not entail its designated answer. The resulting
dataset contains 229 rows.

Important columns include `question`, `chart_answer`, `report_answer`,
`text_report`, `official_table_data`, `source_table`, `chart_image`,
`table_image`, `table_image_sha256`, and the frozen original-manifest hash.

This dataset inherits the use conditions and licensing requirements of ChartQA.
Users should cite the original ChartQA paper and dataset.
