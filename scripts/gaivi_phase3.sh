#!/bin/bash -l
# ── Phase 3: Multi-Benchmark Evaluation on GAIVI ──
# Runs SVAMP (Protocol A), MathVista (Protocol B), ChartQA (Protocol B)
# on 3 models: InternVL2-8B, Qwen2.5-VL-7B, Idefics3-8B
#
# Transformers version conflict:
#   - InternVL2 needs 4.45.2
#   - Qwen2.5-VL and Idefics3 need 5.x
#
# Usage:
#   cd ~/vlm-modality-research
#   bash scripts/gaivi_phase3.sh [batch1|batch2]
#
#   batch1: InternVL2 (transformers 4.45.2) — run while Phi-3.5 is still going
#   batch2: Qwen2.5-VL + Idefics3 (transformers 5.x) — run after upgrading
#
# Monitor: squeue -u $USER

PARTITION="CISL"
REPO_DIR="$HOME/vlm-modality-research"
OUTPUT_DIR="$HOME/vlm_research_results/phase3"
BENCHMARKS="svamp,mathvista,chartqa"

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs"

BATCH="${1:-batch1}"

if [ "$BATCH" == "batch1" ]; then
    echo "=== BATCH 1: InternVL2-8B (transformers 4.45.2) ==="
    MODELS=("OpenGVLab/InternVL2-8B")
elif [ "$BATCH" == "batch2" ]; then
    echo "=== BATCH 2: Qwen2.5-VL-7B + Idefics3-8B (transformers 5.x) ==="
    MODELS=(
        "Qwen/Qwen2.5-VL-7B-Instruct"
        "HuggingFaceM4/Idefics3-8B-Llama3"
    )
else
    echo "Usage: bash scripts/gaivi_phase3.sh [batch1|batch2]"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev)

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
echo "  Phase 3: Multi-Benchmark"
echo "  Model: ${MODEL}"
echo "  Benchmarks: ${BENCHMARKS}"
echo "  Node:  \$(hostname)"
echo "  GPU:   \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Date:  \$(date)"
echo "  Transformers: \$(python -c 'import transformers; print(transformers.__version__)')"
echo "============================================"

cd ${REPO_DIR}

srun python scripts/run_multi_benchmark.py \\
    --config configs/gaivi.yaml \\
    --models "${MODEL}" \\
    --benchmarks "${BENCHMARKS}" \\
    --output-dir "${OUTPUT_DIR}"

echo "Done: ${MODEL} Phase 3 at \$(date)"
SCRIPT

    echo "  Submitted: $SHORT"
done

echo ""
echo "Phase 3 jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Results: $OUTPUT_DIR"
