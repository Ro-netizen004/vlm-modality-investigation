# results/

Experiment outputs, one folder per phase (numbering matches `docs/CANONICAL.md`
and the paper).

```
results/
├── phase1/                     GSM8K main runs
├── phase3/                     Multi-benchmark (Protocol A/B)
├── phase4/                     Noise ablation
├── phase6_legibility/          Legibility — IMAGE arm  (image degraded, text clean)
├── phase7_text_legibility/     Legibility — TEXT arm / mirror (text degraded, image clean)
└── phase6_legibility_ARCHIVED_old/   old local reorg of phase 6 — kept for reference, safe to delete
```

## Legibility arms (phase 6 & 7) — shared structure

Both arms run the **mismatch** conflict (image of problem *i* + text of problem *i+1*)
across a 4-level degradation ladder, and store results identically as
`<benchmark>/<model>/`:

```
phase<N>_.../<benchmark>/<model>/
    level_<L>_<name>.json          per-level summary (text preference, counts)
    level_<L>_<name>.csv           per-trial predictions + `follows` label
    level_<L>_<name>.cll.jsonl     CLL arbitration margin (open models only)
    level_<L>_<name>.logprobs.jsonl  token logprobs (API models only)
    legibility_summary.json        per-model curve across levels
<benchmark>/legibility_all.json    cross-model summary (where present)
```

- **`<benchmark>`** = `gsm8k` or `svamp`.
- **CLL is stored inline** next to each level's `.json`/`.csv` — there is **no**
  separate `cll_gaivi/` folder (that was a transient local convention; removed).

### Which arm degrades what

| Phase | Folder | Image | Text | Level names (`<name>`) |
|-------|--------|-------|------|------------------------|
| 6 | `phase6_legibility/` | **degraded** at level L | clean | `clean`, `blur_light`, `blur_noise`, `heavy_degradation` |
| 7 | `phase7_text_legibility/` | clean | **degraded** at level L | `clean`, `light_corruption`, `medium_corruption`, `heavy_corruption` |

Levels map L0/L2/L4/L5 = clean → light → medium → heavy. The arms use different
`<name>` tokens (blur vs corruption), so files never collide.

### File types

| Suffix | Written by | Contents |
|--------|-----------|----------|
| `.json` | generation (binary) | per-level counts + text preference |
| `.csv` | generation (binary) | one row per trial: both questions, both answers, prediction, `follows` |
| `.cll.jsonl` | `--score-cll` | per-trial `margin` = CLL(text) − CLL(image), joined with `follows`/`reasoning` |
| `.logprobs.jsonl` | API models | per-token logprobs (GPT/Gemini) |
| `legibility_summary.json` | merge | `text_preference_by_level`, `neither_rate_by_level` |

CLL exists only for the 6 forward-pass open models (qwen / llava / llava_onevision /
phi / idefics). GPT-5.6-Luna, InternVL2, and MiniCPM have **no** `.cll.jsonl` by design.

### Completeness (as of last sync)

**Phase 6 (image arm)**
- `gsm8k`: complete for 8/9 models (binary + CLL). **MiniCPM-V-2_6 has only L0** (L2/L4/L5 pending).
- `svamp`: binary complete only for InternVL2 & Phi-3.5; other CLL models have **`.cll.jsonl` only** (binary generation not run). MiniCPM svamp absent.

**Phase 7 (text arm / mirror)**
- `svamp`: **complete for all 6 models** (binary + CLL).
- `gsm8k`: **only Qwen2-VL-2B complete**; the other 5 models need a re-run.

### Reproduce / extend

```bash
# image arm
python scripts/run_legibility.py --benchmark <bm> --models <M> --noise-levels 0 2 4 5
python scripts/run_legibility.py --benchmark <bm> --models <M> --noise-levels 0 2 4 5 --score-cll
# text arm (mirror) — add --channel text
python scripts/run_legibility.py --channel text --benchmark <bm> --models <M> --noise-levels 0 2 4 5
python scripts/run_legibility.py --channel text --benchmark <bm> --models <M> --noise-levels 0 2 4 5 --score-cll
```
Run generation **before** `--score-cll` so the CLL join has the generation CSV.

---

## Backup policy

| Asset | GitHub | Google Drive | HuggingFace |
|-------|--------|--------------|-------------|
| Images | — | — | HF datasets (canonical) |
| Phases 1–3 CSVs | yes (committed) | `vlm_research_results/results_csvs/` | — |
| Phase 4 JSON summaries | yes | optional mirror | — |
| Phase 6 `level_*.csv` | **no** (gitignored) | **yes** (required) | — |
| Phase 6 JSON summaries | yes | optional | — |

**Sync all CSVs to Drive:**

```powershell
powershell -File scripts/sync_results_csvs_to_drive.ps1
```

If Google Drive for Desktop is not installed, upload `vlm_results_csvs.zip` (repo root)
to `My Drive/vlm_research_results/` via [drive.google.com](https://drive.google.com), then unzip there.

Do not mix pilot outputs (n=100) with full-study results in papers or plots.
