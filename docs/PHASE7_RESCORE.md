# Phase 7 (mirror arm) — channel-aware reasoning rescore

## What the rescore does

Mismatch trials pair the **image** of problem *i* with the **text** of problem *i+1*.
Scoring is two-stage:

1. **Stage 1 — answer match** (`score_mismatch_follows`, `src/evaluation.py`): the
   prediction's numeric answer is compared to each problem's gold answer →
   `image` / `text` / `neither` / `ambiguous` / `invalid`. Text preference is reported
   over decidable (`image`+`text`) trials.
2. **Stage 2 — reasoning-trace rescore** (`score_by_reasoning`, applied only to `neither`
   rows): unique keywords (weighted 2×) and unique numbers of each problem are matched
   against the model's chain-of-thought; the trial is reclassified `text_reasoning` /
   `image_reasoning` if one side scores strictly higher.

Final `text_preference = (text + text_reasoning) / (all decidable + reasoning-rescued)`.

## The confound (fixed 2026-07-20)

Phase 7 (mirror arm) holds the **image clean** and **corrupts the text**. The model reads
a corrupted question (`degrade_text`, `src/text_noise.py`), but the results CSV stores the
**clean** `text_question` (`_write_level_csv`). The original rescore — copied from Phase 6
(image channel) — was **not channel-aware**: it matched the reasoning trace against the
clean text.

At high corruption the model's trace echoes the *degraded* tokens (`"the numbers are 15 and
zO"`), which do not match the clean keywords/numbers (`15`, `20`). Genuine text-following in
the `neither` bucket was therefore **undercounted**, biasing rescued preference toward the
image as corruption increased — an evaluator-side artifact pointing in the *same* direction
as the real mirror effect.

## The fix

`rescore_level_from_csv(csv_path, channel, level)` now, for `channel="text"`, reconstructs
the exact corrupted string the model saw before matching:

```python
seed = (problem_id + 1) % n          # == txt_idx used at generation time
text_q = degrade_text(text_q, level, seed=seed)   # deterministic; reproduces the shown text
```

`degrade_text` is deterministic given `(text, level, seed)`, so the corrupted prompt is
reconstructed exactly with **no re-inference**. The image is held clean in this arm, so
`image_question` is left untouched. Phase 6 (`channel="image"`) is unaffected: `corrupt`
stays `False` and output is byte-identical (plus a `"rescore_channel"` provenance field).

Threaded through `rescore_model(...)` and `run_legibility.py --rescore` (honors `--channel`).

## Effect on the results

The change is **modest and mixed** — confirming the confound was real but small and did
**not** manufacture the mirror shift. GSM8K rescored text-preference at L5 (heavy):

| model | raw (stage-1) | clean-text rescore (old) | channel-aware (fixed) |
|---|---|---|---|
| GPT-5.6-Luna | 0.001 | 0.002 | 0.001 |
| Qwen2.5-VL-7B | 0.048 | 0.300 | 0.273 |
| Qwen2-VL-2B | 0.073 | 0.313 | 0.280 |
| llava-onevision | 0.214 | 0.604 | 0.543 |
| Idefics3 | 0.247 | 0.659 | 0.660 |
| llava-v1.6 | 0.419 | 0.504 | 0.510 |
| Phi-3.5 | 0.423 | 0.490 | 0.305 |

**The raw (stage-1-only) trajectory already collapses** for every model, so the mirror
shift exists in pure answer-following before any rescore — the strongest defense against
"is the decline a rescoring artifact?".

The paper's mirror-arm headline (`tab:mirror_cll`) is the **CLL arbitration margin $m$**
(forward passes, `.cll.jsonl`), which the rescore does not touch — **unchanged**. The only
paper number affected is one illustrative parenthetical: Qwen2.5-VL-7B SVAMP behavioral
preference $0.99\to0.33 \Rightarrow 0.99\to0.29$ (channel-aware; strengthens the collapse).

## Paper text (added to `paper/main.tex`)

**Methods (rescore definition, C3):**

> In the text-degradation arm, this rescore matches against the *corrupted* text actually
> presented to the model rather than the canonical prompt, since the reasoning trace echoes
> the degraded tokens; matching the clean prompt would systematically undercount
> text-following at high corruption. We report both the raw answer-following preference and
> the reasoning-rescored preference.

**Footnote (mirror-arm results):**

> This mirror shift is not an artifact of the reasoning rescore. Raw answer-following
> preference already collapses under text corruption (e.g. Qwen2.5-VL-7B 0.99→0.07 on SVAMP,
> 0.97→0.05 on GSM8K at L5, before any rescore), and the corrupted-text-aware rescore —
> which cannot suffer the clean-prompt undercount — reproduces the same trend.

## Regenerating

```bash
# per-level *_rescored.json (channel-aware; --force to overwrite stale files)
python scratchpad/gen_rescored.py --force
# per-model legibility_summary_rescored.json (canonical rescore_model)
python scratchpad/gen_summaries.py
# combined legibility_all[_rescored].json per arm
python scripts/build_legibility_all.py
```

Both `legibility_all_rescored.json` files carry `text_preference_by_level` (channel-aware)
and `text_preference_raw_by_level` (stage-1) side by side.
