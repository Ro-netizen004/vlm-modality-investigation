# phase_control — matched-legibility control analysis

A self-contained analysis that answers the reviewer objection to the Phase 6/7 legibility
asymmetry ("image barely moves arbitration, text collapses it"): **the two corruption
ladders may not remove equivalent information.** Rather than recalibrate + rerun the whole
grid to matched severity (expensive), we **measure** each channel's information loss two
independent ways and test whether a modality effect survives after controlling for it.

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

## Run (svamp first — its CLL arbitration is complete)

```bash
# 1. legibility axes
#   task_acc (GPU, per model)
for M in Idefics3-8B-Llama3 Phi-3.5-vision-instruct Qwen2-VL-2B-Instruct \
         Qwen2.5-VL-7B-Instruct llava-onevision-qwen2-7b-ov-hf llava-v1.6-mistral-7b-hf; do
  python scripts/measure_legibility_decodability.py --models $M --benchmark svamp --num-problems 300
done
#   survival (CPU; needs tesseract for the image channel)
python scripts/measure_legibility_survival.py --benchmark svamp --num-problems 300

# 2. the control test (runs the interaction regression for each available axis)
python scripts/analyze_legibility_control.py --benchmark svamp --metric cll
```
