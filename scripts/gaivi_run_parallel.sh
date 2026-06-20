#!/bin/bash
# ── Run each model on a separate GPU in parallel ──
# This submits 8 independent SLURM jobs, one per model.
# Each job gets 1 GPU on gpu53.
#
# Usage:
#   bash scripts/gaivi_run_parallel.sh
#
# Results saved to: ~/vlm_research_results/<model-name>/

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$HOME/vlm_research_results"
mkdir -p "$OUTPUT_DIR" logs

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

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vlm-${SHORT:0:15}
#SBATCH --partition=CISL
#SBATCH --nodelist=gpu53
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/${SHORT}_%j.log
#SBATCH --error=logs/${SHORT}_%j.err

cd $REPO_DIR
pip install -q -r requirements.txt 2>/dev/null || true

echo "Running model: $MODEL"
echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

python scripts/run_benchmark.py \\
    --config configs/gaivi.yaml \\
    --models "$MODEL" \\
    --output-dir "$OUTPUT_DIR"

echo "Done: $MODEL"
EOF

    echo "  Submitted: $SHORT"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in: $OUTPUT_DIR"
