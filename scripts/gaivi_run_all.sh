#!/bin/bash -l
#SBATCH --job-name=vlm-bench
#SBATCH -p CISL
#SBATCH -w GPU53
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-user=aviralgupta@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/vlm_bench_%j.log
#SBATCH --error=logs/vlm_bench_%j.err

# ── VLM Modality Benchmark — GAIVI GPU53 (L40S) ──
# Runs all 8 models SEQUENTIALLY on 1 GPU.
# Expected runtime: ~3-4 hours total.
#
# Usage:
#   cd ~/vlm-modality-research
#   sbatch scripts/gaivi_run_all.sh
#
# NOTE: If 'CISL' is not your partition name, check with:
#   sinfo
# and update the -p flag above.

conda activate vlm

REPO_DIR="$HOME/vlm-modality-research"
OUTPUT_DIR="$HOME/vlm_research_results"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  VLM Modality Benchmark — GAIVI"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
nvidia-smi

cd "$REPO_DIR"

srun python scripts/run_benchmark.py \
    --config configs/gaivi.yaml \
    --output-dir "$OUTPUT_DIR" \
    --hf-images

echo ""
echo "============================================"
echo "  DONE — Results in: $OUTPUT_DIR"
echo "  $(date)"
echo "============================================"
