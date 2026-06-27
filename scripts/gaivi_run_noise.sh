#!/bin/bash -l
#SBATCH --job-name=vlm-noise
#SBATCH -p CISL
#SBATCH -w GPU53
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-user=rg21@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/vlm_noise_%j.log
#SBATCH --error=logs/vlm_noise_%j.err

# ── Phase 4: Noise Ablation — GAIVI GPU53 ──
# Runs the rendered-image condition across 10 noise levels on GSM8K,
# for the requested models, plus a text-only baseline.
#
# Usage:
#   cd ~/vlm-modality-research
#   sbatch scripts/gaivi_run_noise.sh
#
# NOTE: If 'CISL' is not your partition, check with `sinfo` and update -p above.
# Keep model cache off /home (100GB limit) — set HF_HOME to /data or /general.

conda activate vlm

REPO_DIR="$HOME/vlm-modality-investigation"
OUTPUT_DIR="$HOME/vlm_research_results/phase4"
export HF_HOME="${HF_HOME:-/data/rg21/hf_cache}"

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs" "$HF_HOME"

echo "============================================"
echo "  Phase 4: Noise Ablation — GAIVI"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "  Output: $OUTPUT_DIR"
echo "  HF_HOME: $HF_HOME"
echo "============================================"
nvidia-smi

cd "$REPO_DIR"

# 7 models (Phi-3.5 excluded for now), full 10 noise levels, 200-problem subset.
# To add Phi back: append "Phi-3.5-vision-instruct" to the --models list.
srun python scripts/run_noise_ablation.py \
    --models Qwen2-VL-2B-Instruct llava-v1.6-mistral-7b-hf Qwen2.5-VL-7B-Instruct \
             Idefics3-8B-Llama3 MiniCPM-V-2_6 InternVL2-8B \
             llava-onevision-qwen2-7b-ov-hf \
    --num-problems 200 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "  DONE — Results in: $OUTPUT_DIR"
echo "  $(date)"
echo "============================================"
