---
pretty_name: ChartQA Evidence Conflict
task_categories:
  - visual-question-answering
language:
  - en
size_categories:
  - n<1K
---

# ChartQA Evidence Conflict

This dataset contains 230 reviewed conflicts between a native ChartQA chart and
an evidence-bearing textual report. The original chart supports one answer,
while counterfactual facts in the report support a distinct answer to the same
question. Neither source is privileged in the evaluation prompt.

Each row contains the original chart, shared question, chart-supported answer,
report-supported answer, evidence-bearing report, unit and counterfactual
strategy, counterbalanced source labels, original ChartQA test index,
source-table provenance, and review metadata.

The frozen manifest SHA-256 is
`388bce0572487024f5ac12261621cbab8931ec3032d8bf0c65a258c134d20842`.
All rows use design version `chartqa-same-question-conflict-v4`.

## Intended evaluation

Generated responses are attributed using strict normalized exact matching:
chart answer, report answer, neither, ambiguous, or invalid. Do not use fuzzy
matching or lexical reasoning-trace attribution. The chart and report can be
degraded independently at matched experimental levels; degradation variants
are produced deterministically at evaluation time and are not stored here.

## Source and provenance

Charts and original questions/answers come from
[`lmms-lab/ChartQA`](https://huggingface.co/datasets/lmms-lab/ChartQA).
Counterfactual reports were manually reviewed for entailment, unit consistency,
and answer distinctness. `source_table` identifies the official ChartQA table
used during curation.

This derivative dataset does not change the original charts. Users remain
responsible for complying with the upstream ChartQA dataset terms and citation
requirements.
