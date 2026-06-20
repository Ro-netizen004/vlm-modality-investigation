#!/bin/bash
#SBATCH --job-name=vlm-bench
#SBATCH --partition=CISL
#SBATCH --nodelist=gpu53
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vlm_bench_%j.log
#SBATCH --error=logs/vlm_bench_%j.err

# ── VLM Modality Benchmark — GAIVI GPU53 (L40S) ──
# Usage:
#   sbatch scripts/gaivi_run_all.sh                    # all models
#   sbatch scripts/gaivi_run_all.sh --models "Qwen/Qwen2-VL-2B-Instruct"
#
# Results saved to: ~/vlm_research_results/

set -euo pipefail

# ── Setup ──
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$HOME/vlm_research_results"
mkdir -p "$OUTPUT_DIR" logs

echo "============================================"
echo "  VLM Modality Benchmark — GAIVI"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "  Date: $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

nvidia-smi

# ── Environment ──
cd "$REPO_DIR"

# Use existing conda/venv if available, otherwise install deps
if command -v conda &>/dev/null; then
    # If a vlm environment exists, activate it
    conda activate vlm 2>/dev/null || true
fi

pip install -q -r requirements.txt 2>/dev/null || true

# ── Run benchmark ──
python scripts/run_benchmark.py \
    --config configs/gaivi.yaml \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "============================================"
echo "  DONE — Results in: $OUTPUT_DIR"
echo "  $(date)"
echo "============================================"
