#!/bin/bash -l
# ── Phase 3 Expanded: All benchmarks × all models ──
# Submits jobs for models compatible with current transformers version.
#
# With transformers 4.45.2 (batch_a):
#   - InternVL2-8B: remaining 4 benchmarks (aqua_rat, math, scienceqa, ai2d)
#   - LLaVA-1.6-7B: all 7 benchmarks
#   - MiniCPM-V-2.6: all 7 benchmarks
#
# With transformers 5.x (batch_b) — run after Phi-3.5 finishes:
#   - Qwen2.5-VL-7B: all 7 benchmarks
#   - Idefics3-8B: all 7 benchmarks
#   - LLaVA-OneVision-7B: all 7 benchmarks
#   - Qwen2-VL-2B: all 7 benchmarks
#
# Usage:
#   cd ~/vlm-modality-research
#   bash scripts/gaivi_phase3_expanded.sh batch_a
#   bash scripts/gaivi_phase3_expanded.sh batch_b

PARTITION="CISL"
REPO_DIR="$HOME/vlm-modality-research"
OUTPUT_DIR="$HOME/vlm_research_results/phase3"
ALL_BENCHMARKS="svamp,aqua_rat,math,mathvista,scienceqa,ai2d,chartqa"

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs"

BATCH="${1:-batch_a}"

submit_job() {
    local MODEL="$1"
    local BENCHMARKS="$2"
    local SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev)

    sbatch <<SCRIPT
#!/bin/bash -l
#SBATCH --job-name=p3-${SHORT:0:12}
#SBATCH -p ${PARTITION}
#SBATCH -w GPU53
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=aviralgupta@usf.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/phase3_${SHORT}_%j.log
#SBATCH --error=logs/phase3_${SHORT}_%j.err

conda activate vlm

echo "============================================"
echo "  Phase 3 Expanded"
echo "  Model: ${MODEL}"
echo "  Benchmarks: ${BENCHMARKS}"
echo "  Node:  \$(hostname)"
echo "  GPU:   \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Date:  \$(date)"
echo "  Transformers: \$(python -c 'import transformers; print(transformers.__version__)')"
echo "============================================"

cd ${REPO_DIR}

srun python scripts/run_multi_benchmark.py \
    --config configs/gaivi.yaml \
    --models "${MODEL}" \
    --benchmarks "${BENCHMARKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --hf-images

echo "Done: ${MODEL} at \$(date)"
SCRIPT

    echo "  Submitted: $SHORT ($BENCHMARKS)"
}

if [ "$BATCH" == "batch_a" ]; then
    echo "=== BATCH A: transformers 4.45.2 models ==="
    echo ""
    # InternVL2 already has svamp, mathvista, chartqa — run remaining 4
    submit_job "OpenGVLab/InternVL2-8B" "aqua_rat,math,scienceqa,ai2d"
    # LLaVA-1.6 and MiniCPM need all 7
    submit_job "llava-hf/llava-v1.6-mistral-7b-hf" "$ALL_BENCHMARKS"
    submit_job "openbmb/MiniCPM-V-2_6" "$ALL_BENCHMARKS"

elif [ "$BATCH" == "batch_b" ]; then
    echo "=== BATCH B: transformers 5.x models ==="
    echo ""
    submit_job "Qwen/Qwen2.5-VL-7B-Instruct" "$ALL_BENCHMARKS"
    submit_job "HuggingFaceM4/Idefics3-8B-Llama3" "$ALL_BENCHMARKS"
    submit_job "llava-hf/llava-onevision-qwen2-7b-ov-hf" "$ALL_BENCHMARKS"
    submit_job "Qwen/Qwen2-VL-2B-Instruct" "$ALL_BENCHMARKS"

else
    echo "Usage: bash scripts/gaivi_phase3_expanded.sh [batch_a|batch_b]"
    exit 1
fi

echo ""
echo "Jobs submitted! Monitor with: squeue -u \$USER"
echo "Results: $OUTPUT_DIR"
