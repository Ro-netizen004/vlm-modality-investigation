# results/

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
