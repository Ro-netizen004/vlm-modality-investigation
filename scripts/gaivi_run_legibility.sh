#!/bin/bash -l
#SBATCH --job-name=vlm-legib
#SBATCH -p CISL
#SBATCH -w GPU53
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-user=rg21@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/vlm_legib_%j.log
#SBATCH --error=logs/vlm_legib_%j.err

# ── Legibility experiment — mismatch condition x noise levels ──
# Measures whether modality (text) preference shifts as the image degrades.
# Reuses the Phase 4 noisy images (results/phase4/images) if present.
#
# Usage:
#   cd ~/vlm-modality-investigation
#   sbatch scripts/gaivi_run_legibility.sh

conda activate vlm

REPO_DIR="$HOME/vlm-modality-investigation"
OUTPUT_DIR="$HOME/vlm_research_results/phase6_legibility"
NOISE_IMAGES="$HOME/vlm_research_results/phase4/images"
export HF_HOME="${HF_HOME:-/data/rg21/hf_cache}"

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs" "$HF_HOME"

echo "============================================"
echo "  Legibility experiment — GAIVI"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
nvidia-smi

cd "$REPO_DIR"

# Paper-scale anchors (vulnerable + resilient). Default N=300 in run_legibility.py;
# stale level_*.json from an old N are auto-cleared on rerun.
srun python scripts/run_legibility.py \
    --models Qwen2.5-VL-7B-Instruct Idefics3-8B-Llama3 \
    --num-problems 300 \
    --noise-image-dir "$NOISE_IMAGES" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "  DONE — Results in: $OUTPUT_DIR"
echo "  $(date)"
echo "============================================"
