# phase_control — control & robustness analyses

Self-contained analyses that answer the two most dangerous reviewer objections to the
Phase 6/7 legibility asymmetry, without re-running the main grid.

**(A) Matched-legibility control** — "the two corruption ladders may not remove equivalent
information, so the 6-vs-2 asymmetry could be a severity artifact." We **measure** each
channel's information loss two independent ways and test whether a modality effect survives
after controlling for it (`measure_legibility_*` + `analyze_legibility_control.py`).

**(B) Visual-reliance probe** — "rendered text ≠ real visual reasoning; your image findings
may be about OCR, not vision." We test the *same* reliability question on genuinely visual
content (charts/diagrams) (`measure_visual_reliance.py`). See the section near the bottom.

## Two legibility axes (triangulation)

| Axis | Script | Measures | Cost |
|------|--------|----------|------|
| **task_acc** | `scripts/measure_legibility_decodability.py` | single-modality VLM accuracy (image-only / text-only) — what the model can **exploit** | GPU, per-model |
| **survival** | `scripts/measure_legibility_survival.py` | OCR (Tesseract) / char survival — **perceptual** information available (model-independent) | CPU; text side is free/deterministic |

Agreement between them is a strong robustness result (perception vs exploitation tell the
same story); divergence localizes failure to one or the other.

## The test — `scripts/analyze_legibility_control.py`

Fits, for **each** legibility axis, over points (channel × level × model):

```
arb_shift ~ b0 + b1*leg_loss + b2*modality + b3*(modality x leg_loss)
```
`leg_loss` = fractional legibility loss of the degraded channel `(L0-L)/L0`;
`modality` = 0 image / 1 text; `arb_shift` = shift toward the clean channel (CLL margin by default).

- **b3 (interaction) ≈ 0, n.s.** → the raw 6-vs-2 asymmetry is explained by legibility loss:
  controlling for information lost, VLMs weight both channels by reliability (rational).
- **b3 > 0, sig.** → residual **text-primacy** beyond legibility loss (asymmetry survives).

Reporting the same b3 verdict across **both** axes = the reviewer-proof statement.
A **headroom guard** (`--min-headroom`) drops task_acc points whose L0 accuracy is near chance
(no room to lose → noisy fractional loss); survival always starts at 1.0 so it is unaffected.

## Layout

```
results/phase_control/
├── decodability/<benchmark>/decodability_all.json   # task_acc axis (per model)
└── survival/<benchmark>/survival.json               # survival axis (model-independent)
```
Arbitration is read from `results/phase6_legibility/<bm>/<model>/` (image arm) and
`results/phase7_text_legibility/<bm>/<model>/` (text arm) — no duplication.

## N per axis (why they differ)

| Axis | svamp | gsm8k | why |
|------|-------|-------|-----|
| **survival** (OCR/CER) | 300 | **1319** | cheap (CPU/free) → match the arbitration N, and future-proof a per-trial control |
| **task_acc** (VLM accuracy) | 300 | 300 | GPU per-model; a per-level accuracy is stable at 300 (±~5%), 1319 = 4.4x compute for ~0 gain |

Match survival to the arbitration N (svamp=300, gsm8k=1319). Keep task_acc at 300 both.

## Run (svamp first — its CLL arbitration is complete)

**task_acc** (GPU, per model — runs in the `vlm` env alongside the grid):
```bash
for BM_N in "svamp 300" "gsm8k 300"; do set -- $BM_N; BM=$1; N=$2
  for M in Idefics3-8B-Llama3 Phi-3.5-vision-instruct Qwen2-VL-2B-Instruct \
           Qwen2.5-VL-7B-Instruct llava-onevision-qwen2-7b-ov-hf llava-v1.6-mistral-7b-hf; do
    python scripts/measure_legibility_decodability.py --models $M --benchmark $BM --num-problems $N
  done
done
```
Use `--channels text` or `--channels image` for a partial grid. The runner resumes
completed model/level cells from the per-model JSON and rebuilds the aggregate from all
compatible per-model files, so separate one-model Slurm jobs do not erase prior channels.

**survival** (CPU, model-independent). Use an ISOLATED conda env so the tesseract install
never perturbs the `vlm` env your running/queued jobs depend on:
```bash
# NB: let pip provide numpy — conda-forge numpy is built for x86-64-v2 and FAILS to import
# on the older gaivi-login1 CPU ("baseline optimizations (X86_V2)..."). pip wheels use the
# generic baseline and run on the login node.
conda create -y -n ocr -c conda-forge python=3.11 tesseract pillow
conda activate ocr && pip install pytesseract datasets tqdm numpy
export HF_HOME=/data/rg21/hf_cache
python scripts/measure_legibility_survival.py --benchmark svamp --num-problems 300
python scripts/measure_legibility_survival.py --benchmark gsm8k --num-problems 1319
conda deactivate   # (conda env remove -n ocr when done)
```
Do **not** `conda install tesseract` into `vlm` while the grid is queued — it can bump shared
packages and every pending job would inherit the change. If numpy still errors on the login
node, run the survival pass on a compute node: `srun -p Quick,CISL -c 4 --mem=16G -t 00:30:00 --pty bash`.

## The control test (interaction regression for each available axis)
```bash
python scripts/analyze_legibility_control.py --benchmark svamp --metric cll
python scripts/analyze_legibility_control.py --benchmark gsm8k --metric cll
```

---

# (B) Visual-reliance probe — `scripts/measure_visual_reliance.py`

Answers the ecological-validity objection: *rendered text ≠ real visual reasoning; the image
findings may be about OCR, not vision.* We run the **same reliability question on genuinely
visual content** (charts/diagrams). No conflict construction needed.

On a vision-essential benchmark (ChartQA / AI2D), present image+question at each image
degradation level (0/2/4/5) and measure per level:
- **accuracy(L)** — falls if the image is essential and being read
- **confidence(L)** — mean generated-token logprob; a reliability-aware model loses confidence as accuracy drops
- **invariance(L)** — fraction of answers unchanged from L0 (high under heavy blur ⇒ the model ignores the degraded image)

**Read-out (matches the rendered-text finding if):** accuracy collapses while confidence stays
flat and/or answers stay invariant ⇒ the model keeps confidently answering a chart it can no
longer read ⇒ reliability insensitivity **generalizes beyond OCR**. If instead confidence /
answer-change track the accuracy drop, models *are* reliability-aware for pictorial content
and the rendered-text result is OCR-specific — either way, an honest, publishable answer.

**ChartQA is the strongest choice** (text-only accuracy ≈ 0, so the image is maximally essential).

```bash
# GPU, per model (queues behind the grid); MiniCPM excluded (version-broken)
for BM in chartqa ai2d; do
  for M in Qwen2.5-VL-7B-Instruct InternVL2-8B Idefics3-8B-Llama3 Phi-3.5-vision-instruct \
           Qwen2-VL-2B-Instruct llava-onevision-qwen2-7b-ov-hf llava-v1.6-mistral-7b-hf; do
    sbatch -p Quick,CISL --gpus=1 -c 8 --mem=64G -t 04:00:00 \
      -o logs/visrel_${BM}_${M}_%j.log \
      --wrap="source ~/.bashrc; conda activate vlm; export HF_HOME=/data/rg21/hf_cache; cd ~/vlm-modality-investigation && python scripts/measure_visual_reliance.py --models ${M} --benchmark ${BM} --num-problems 300"
  done
done
```
Writes `results/phase_control/visual_reliance/<bm>/`; the script prints a per-model
`acc L0->L5 | conf | invariance@L5` trajectory read.
