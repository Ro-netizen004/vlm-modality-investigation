# PROJECT STATUS / HANDOFF — VLM modality-arbitration study

Living status doc so a fresh session (human or AI) can pick up without re-deriving context.
**Read this + `docs/CANONICAL.md` first.** Update the "Experiment status" table when runs land.

Last substantive update: 2026-07-20.

---

## 1. What the paper is

Do VLMs arbitrate between conflicting **text** and **image** inputs like a reliability-weighted
observer (down-weight the degraded channel), or follow a fixed prior? Method: a **mismatch**
conflict (image of problem *i* + text of problem *i+1*) where the model's modality choice is
observable, degraded **symmetrically** — image arm (Phase 6) vs text arm / mirror (Phase 7).

**Headline so far:** a stark **asymmetry** — degrade the *image* and only ~2/6 models respond;
degrade the *text* and **6/6** collapse toward the image (SVAMP; GSM8K text now complete too).
Reads as a **text-primary process**, not balanced cue integration. Target venue: **EACL** (main
is a stretch; Findings realistic). Two graded measures: behavioral text-preference, and the
**CLL arbitration margin** (`margin = CLL(text_ans) − CLL(image_ans)`, open models only).

Paper source: `paper/main.tex` (**gitignored** — stays local). §4.8 has a scaffold for the mirror
arm awaiting final both-benchmark numbers.

---

## 2. Experiment status (update me)

Legend: ✅ complete · 🟡 partial/running · ❌ missing · N = problems.

| Arm / benchmark | Binary (behavioral) | CLL margin | N | Notes |
|---|---|---|---|---|
| **P6 image · gsm8k** | ✅ 8 open + Luna (MiniCPM L0 only) | ✅ 6 CLL models | 1319 | rescore done |
| **P6 image · svamp** | 🟡 InternVL2+Phi done; **5 open RUNNING** | ✅ 6 (missing follows/reasoning join) | 300 | binary loop submitted this session |
| **P7 text · gsm8k** | ✅ 6 CLL + Luna | ✅ 6 CLL | 1319 | InternVL2 binary RUNNING |
| **P7 text · svamp** | ✅ 6 CLL | ✅ 6 CLL | 300 | InternVL2 binary RUNNING |

**Frontier = GSM8K only** (budget): GPT-5.6-Luna done on both arms of gsm8k. No SVAMP frontier.
GPT-4o-mini registered but **not run** (would be gsm8k-only if pursued — gives answer-confidence
trajectory via logprobs, NOT a CLL margin).

**phase_control (robustness):** `survival` axis (OCR/CER) done for gsm8k+svamp; `decodability`
(task-acc) and `visual_reliance` **not run yet**.

**Known-missing / TODO:** InternVL2 text arm (running); MiniCPM everywhere (version-broken);
SVAMP-image CLL follows/reasoning join (recoverable once the binary CSVs land).

---

## 3. GAIVI cluster essentials

- **Login:** `ssh rg21@gaivi.cse.usf.edu` → host `gaivi-login1`. Interactive GPU: `srun -p Quick,CISL --gpus=1 -c 8 --mem=64G -t 00:30:00 --pty bash`.
- **Repo on cluster:** `~/vlm-modality-investigation` (NOT `vlm-modality-research` — the GitHub repo is `Ro-netizen004/vlm-modality-investigation`).
- **Env:** `conda activate vlm` (Python 3.10, **transformers pinned 4.49.0** — see §5 MiniCPM).
- **HF cache:** `export HF_HOME=/data/rg21/hf_cache` (must be set or weights fill `$HOME`).
- **Results (runtime outputs):** `~/vlm_research_results/` — `phase6_legibility/` (image arm),
  `phase7_mirror/` (text arm). **phase_control outputs go to the REPO dir**: `~/vlm-modality-investigation/results/phase_control/`.
- **Partitions:** `Quick,CISL`. **QOS cap ≈ 6 concurrent jobs** → large fan-outs run in waves
  (`squeue` shows `PD ... (QOSMaxJobsPerUserLimit)` — normal).

### Submission pattern that WORKS (learned the hard way)
```bash
cd ~/vlm-modality-investigation            # submit FROM the repo dir
LOGS=$HOME/vlm-modality-investigation/logs; mkdir -p "$LOGS"
sbatch -p Quick,CISL --gpus=1 -c 8 --mem=64G -t 06:00:00 \
  -o "$LOGS/<name>_%j.log" \               # ABSOLUTE -o path
  --wrap="source ~/.bashrc; conda activate vlm; export HF_HOME=/data/rg21/hf_cache; cd ~/vlm-modality-investigation && python scripts/run_legibility.py ..."
```
- Chain generation `&&` `--score-cll` in ONE `--wrap` so CLL gets the generation CSV to join.
- **Gotcha:** submitting from `$HOME` with a *relative* `-o logs/...` → Slurm can't create the
  output file → job dies in ~1s (`ExitCode 0:53`, no log written). Use the pattern above.

### Monitoring / diagnosis
```bash
squeue -u rg21
sacct -u rg21 --starttime today --format=JobID,JobName%22,State,Elapsed,ExitCode | tail
tail -f "$(ls -t logs/<name>_*.log | head -1)"
```

---

## 4. Cluster → local layout (they DIFFER — reorganize on pull)

**Cluster (run_legibility output convention):**
- gsm8k image → `phase6_legibility/<model>/` (flat); svamp image → `phase6_legibility/svamp/<model>/`
- gsm8k text → `phase7_mirror/text_legibility/<model>/` (flat); svamp text → `phase7_mirror/svamp/text_legibility/<model>/`

**Local canonical (reorganized to `<benchmark>/<model>`):**
- `results/phase6_legibility/<benchmark>/<model>/`  (image arm)
- `results/phase7_text_legibility/<benchmark>/<model>/`  (text arm)
- `results/phase_control/{decodability,survival,visual_reliance}/<benchmark>/`

Per cell: `level_<L>_<name>.{json,csv,cll.jsonl,logprobs.jsonl}`, `legibility_summary.json`,
`level_*_rescored.json`. Level names differ by arm: image = `clean/blur_light/blur_noise/heavy_degradation`;
text = `clean/light/medium/heavy_corruption`. CLL is **inline** (no separate `cll_gaivi/`).

### Pull (PowerShell, scp per-model to skip the sibling `noise_images/`)
```powershell
$base = "rg21@gaivi.cse.usf.edu:~/vlm_research_results/phase7_mirror/text_legibility"
$models = "Idefics3-8B-Llama3","Phi-3.5-vision-instruct","Qwen2-VL-2B-Instruct","Qwen2.5-VL-7B-Instruct","llava-onevision-qwen2-7b-ov-hf","llava-v1.6-mistral-7b-hf","GPT-5.6-Luna"
foreach ($m in $models) { scp -r "$base/$m" "results/phase7_text_legibility/gsm8k/" }
```
`rsync -av --exclude 'noise_images' --exclude '*.logprobs.jsonl' <src>/ <dst>/` is cleaner if you
run it from Git Bash. After pulling, run `python scripts/build_legibility_all.py` to rebuild
`legibility_all.json` / `_rescored.json` for the new tree.

---

## 5. Known issues & gotchas

- **MiniCPM-V-2.6 is version-broken** in the legibility runs (works in Phase 1/3, fails Phase 6/7).
  `requirements.txt` pins `transformers==4.49.0` (>=4.50 drops GenerationMixin → breaks
  InternVL2/MiniCPM; Qwen2.5-VL needs >=4.49, so 4.49.0 is the only compatible pin). InternVL2
  works at 4.49.0 but MiniCPM still fails → root cause likely its unpinned `trust_remote_code`
  remote code, not just the lib version. **Excluded from all legibility runs; weights removed to
  free quota.** Same loader (`src/models.py` `_load_minicpm`) across phases — not a code diff.
- **`/data/rg21` per-user quota ≈ 100 GB, all model weights** (`hf_cache/hub`). Can't hold all 8
  VLMs (~15 GB each) at once → `OSError [Errno 122] Disk quota exceeded` on dataset load. Prune
  models between runs: `rm -rf /data/rg21/hf_cache/hub/models--<org>--<name>`. Check: `du -sh /data/rg21`.
- **Login-node CPU lacks `x86-64-v2`** → conda-forge numpy fails to import
  (`baseline optimizations (X86_V2)`). Use pip's numpy (generic baseline) or run on a compute node.
- **$HOME quota** was also ~100 GB and filled by the HF cache before `HF_HOME` was set — keep
  `HF_HOME=/data/rg21/hf_cache` exported everywhere (`~/.bashrc` has it).
- **Windows/PowerShell:** no `&&`, no `rm -rf`/`mkdir -p`; use `;`, `Remove-Item -Recurse -Force`,
  `New-Item -ItemType Directory -Force`. `run_legibility.py` had `→` prints that crashed cp1252 (fixed to `->`).

---

## 6. Scripts (all in `scripts/`)

| Script | Purpose |
|---|---|
| `run_legibility.py` | **Workhorse.** Mismatch legibility. Flags: `--channel {image,text}`, `--score-cll`, `--benchmark`, `--num-problems`, `--noise-levels`, `--output-dir`, `--render-only`, `--merge`, `--rescore`. `MODEL_REGISTRY` + `CLL_TYPES` (qwen/llava/llava_onevision/phi/idefics = 6 CLL-capable; internvl/minicpm/openai/gemini = binary-only). |
| `build_legibility_all.py` | Collate `legibility_all.json` + `_rescored.json` from the reorganized `<benchmark>/<model>` tree (replaces `--merge`, which assumes old paths). |
| `analyze_cll.py` | CLL analysis (scale validation, ceiling-cracker, dissociation). **`ROOT` is hardcoded to old `cll_gaivi/` image-arm paths** — needs `--root`/path fix for the new layout. |
| `analyze_confidence.py` | Frontier answer-confidence trajectory from `.logprobs.jsonl` (API models; Luna has no logprobs, 4o-mini would). |
| **`measure_legibility_decodability.py`** | phase_control: **task-accuracy** legibility axis (single-modality VLM accuracy, per model, GPU). |
| **`measure_legibility_survival.py`** | phase_control: **OCR/char-survival** legibility axis (model-independent; text=free, image=Tesseract). Run in isolated `ocr` conda env — see `results/phase_control/README.md`. |
| **`analyze_legibility_control.py`** | phase_control: **β3 interaction test** (`arb_shift ~ b1·leg_loss + b2·modality + b3·(mod×leg)`) per legibility axis. Answers "unmatched corruption severity." |
| **`measure_visual_reliance.py`** | phase_control: **Protocol-B probe** (ChartQA/AI2D) — does reliance track legibility on *genuinely visual* content? Answers "rendered text ≠ real visual reasoning." |
| `build_conflict_dataset.py` / `upload_conflict_dataset.py` | Build/publish the released benchmark (both arms). |

---

## 7. Datasets (HuggingFace, `vlm-modality-research` org)

- **Released benchmark:** `modality-conflict-arbitration-v2` (both arms; **private** — make `--public` after review). Card: `docs/CONFLICT_DATASET_CARD.md`.
- **Rendered images:** `gsm8k-rendered-vlm-v2` (900px), `svamp-rendered-vlm-v1`. Do **not** mix v1/v2.

---

## 8. Reviewer defenses (the robustness story)

Two highest-danger objections, each with a built analysis in `results/phase_control/`
(see its README):
1. **"Unmatched corruption severity"** → `measure_legibility_{decodability,survival}` + `analyze_legibility_control` (β3 test on two legibility axes: OCR survival + task accuracy). Agreement across axes = robust.
2. **"Rendered text ≠ real visual reasoning"** (ranked #1 danger) → `measure_visual_reliance` on ChartQA/AI2D: does reliance on a real chart/diagram track its legibility?

Also being pursued for main-track credibility: finish GSM8K text arm (done), both-benchmark
replication, and honest scoping in Limitations.

---

## 9. Immediate next steps

1. Let the **SVAMP image binary (5 open)** + **InternVL2 text (gsm8k L0/2/4/5 + svamp)** jobs finish; pull → `phase6_legibility/svamp/`, `phase7_text_legibility/{gsm8k,svamp}/`; `build_legibility_all.py`.
2. **Verify `survival.json` has the image (OCR) channel**, not just text — needed for the β3 test.
3. Run **GSM8K symmetric CLL analysis** (image vs text arm) — mirror of the SVAMP result; fills §4.8 for both benchmarks.
4. Optional/high-value: **visual-reliance probe** (ChartQA) for the #1 reviewer weakness; **decodability** axis for the control triangulation.
5. Post-hoc **join SVAMP-image CLL** with the new generation CSVs (recover follows/reasoning).
