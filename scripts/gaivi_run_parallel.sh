#!/bin/bash -l
# ── Run each model as a separate SLURM job on gpu53 ──
# Submits 8 independent jobs. Each gets 1 GPU.
# gpu53 has 8× L40S, so all 8 can run simultaneously.
# Expected runtime: ~45-90 min (limited by the slowest model).
#
# Usage:
#   cd ~/vlm-modality-research
#   bash scripts/gaivi_run_parallel.sh
#
# Monitor:  squeue -u $USER
# Cancel:   scancel <jobID>  or  scancel -u $USER
#
# NOTE: If 'CISL' is not your partition name, change PARTITION below.

PARTITION="CISL"
REPO_DIR="$HOME/vlm-modality-research"
OUTPUT_DIR="$HOME/vlm_research_results"
mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs"

MODELS=(
    "Qwen/Qwen2-VL-2B-Instruct"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "OpenGVLab/InternVL2-8B"
    "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    "microsoft/Phi-3.5-vision-instruct"
    "openbmb/MiniCPM-V-2_6"
    "HuggingFaceM4/Idefics3-8B-Llama3"
)

echo "Submitting ${#MODELS[@]} jobs to gpu53..."
echo ""

for MODEL in "${MODELS[@]}"; do
    SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev)

    sbatch <<SCRIPT
#!/bin/bash -l
#SBATCH --job-name=vlm-${SHORT:0:15}
#SBATCH -p ${PARTITION}
#SBATCH -w gpu53
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-user=aviralgupta@usf.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/${SHORT}_%j.log
#SBATCH --error=logs/${SHORT}_%j.err

conda activate vlm

echo "============================================"
echo "  Model: ${MODEL}"
echo "  Node:  \$(hostname)"
echo "  GPU:   \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  Date:  \$(date)"
echo "============================================"

cd ${REPO_DIR}

srun python scripts/run_benchmark.py \\
    --config configs/gaivi.yaml \\
    --models "${MODEL}" \\
    --output-dir "${OUTPUT_DIR}"

echo "Done: ${MODEL} at \$(date)"
SCRIPT

    echo "  Submitted: $SHORT"
done

echo ""
echo "All ${#MODELS[@]} jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Results will appear in: $OUTPUT_DIR"
