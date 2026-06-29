#!/bin/bash -l
#SBATCH --job-name=vlm-mech
#SBATCH -p CISL
#SBATCH -w GPU53
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-user=rg21@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/vlm_mech_%j.log
#SBATCH --error=logs/vlm_mech_%j.err

# ── Phase 6: Mechanistic Analysis — GAIVI GPU53 ──
# Attention-to-image, OCR quality, and hidden-state similarity on a small
# GSM8K subset. Compute-light: 1-2 models in well under an hour (2h limit is
# generous headroom).
#
# Usage:
#   cd ~/vlm-modality-investigation
#   sbatch scripts/gaivi_run_mechanistic.sh

conda activate vlm

REPO_DIR="$HOME/vlm-modality-investigation"
OUTPUT_DIR="$HOME/vlm_research_results/phase6"
export HF_HOME="${HF_HOME:-/data/rg21/hf_cache}"

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs" "$HF_HOME"

echo "============================================"
echo "  Phase 6: Mechanistic Analysis — GAIVI"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
nvidia-smi

cd "$REPO_DIR"

# Start with Qwen2-VL-2B (clean attention access). Add Qwen2.5-VL-7B once the
# pipeline is confirmed working.
srun python scripts/run_mechanistic.py \
    --models Qwen2-VL-2B-Instruct \
    --num-problems 50 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "  DONE — Results in: $OUTPUT_DIR"
echo "  $(date)"
echo "============================================"
