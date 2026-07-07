#!/bin/bash -l
# ── Legibility experiment, fanned out across the cluster ──
# Submits ONE SLURM job per (benchmark, model, noise-level) so they run
# concurrently on the 8x L40S in GPU53, instead of one serial job on a single
# GPU (which hit the 12h wall-time after ~4 levels of a single model).
#
# Grid: 2 benchmarks x 8 models x 4 legibility levels = 64 compute jobs.
# On 8 L40S that's ~8 waves; wall clock ~= 8x the slowest cell (Idefics3).
#
# For each benchmark:
#   1. a short CPU "prep" job applies noise to the CANONICAL HF images ONCE
#      (--render-only), so the compute jobs never race to write the same files
#      AND Level 0 is pixel-identical to the images used in the main experiments;
#   2. the (model x level) compute jobs depend on that prep job;
#   3. a merge job (depends on all compute cells) collates the per-level JSONs.
#
# Usage:
#   cd ~/vlm-modality-research
#   bash scripts/gaivi_run_legibility_parallel.sh
#
# Monitor:  squeue -u $USER      Cancel: scancel -u $USER

PARTITION="CISL"
NODE="GPU53"                                   # 8x L40S
REPO_DIR="$HOME/vlm-modality-investigation"    # <-- your GAIVI checkout dir
OUTPUT_DIR="$HOME/vlm_research_results/phase6_legibility"
export HF_HOME="${HF_HOME:-/data/rg21/hf_cache}"

# Paper-scale default: full SVAMP (300); solid GSM8K subset. run_legibility.py
# auto-invalidates stale level_*.json when --num-problems changes.
NUM_PROBLEMS=300

BENCHMARKS=("gsm8k" "svamp")   # numeric Protocol-A benchmarks
MODELS=(                       # full 8-model set (headline run)
    "Qwen2-VL-2B-Instruct"
    "llava-v1.6-mistral-7b-hf"
    "Qwen2.5-VL-7B-Instruct"
    "Idefics3-8B-Llama3"
    "MiniCPM-V-2_6"
    "InternVL2-8B"
    "llava-onevision-qwen2-7b-ov-hf"
    "Phi-3.5-vision-instruct"
)
LEVELS=(0 2 4 5)   # monotonic legibility ladder: clean -> light blur -> blur+noise -> heavy

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs" "$HF_HOME"

# Idefics3 is slowest (~3h/level, heavy visual-token splitting); MiniCPM/LLaVA/Phi
# are medium; the Qwen and InternVL models are fast. Give each a safe wall-time.
walltime_for() {
    case "$1" in
        Idefics3-8B-Llama3)                                echo "10:00:00" ;;
        MiniCPM-V-2_6|llava-onevision-qwen2-7b-ov-hf|\
        llava-v1.6-mistral-7b-hf|Phi-3.5-vision-instruct)  echo "06:00:00" ;;
        *)                                                 echo "04:00:00" ;;
    esac
}

LEVELS_STR="${LEVELS[*]}"
TOTAL=0
for BM in "${BENCHMARKS[@]}"; do

    # ── 1. Prep job: apply noise to canonical HF images once (all benchmarks) ──
    PREP_OUT=$(sbatch <<SCRIPT
#!/bin/bash -l
#SBATCH --job-name=legib-prep-${BM}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/legib_prep_${BM}_%j.log
#SBATCH --error=logs/legib_prep_${BM}_%j.err

conda activate vlm
export HF_HOME="${HF_HOME}"
cd ${REPO_DIR}
srun python scripts/run_legibility.py --render-only --benchmark ${BM} --num-problems ${NUM_PROBLEMS} --noise-levels 0 2 4 5 --output-dir "${OUTPUT_DIR}"
SCRIPT
)
    PREP_JID=$(echo "$PREP_OUT" | awk '{print $NF}')
    PREP_DEP="--dependency=afterok:${PREP_JID}"
    echo "[${BM}] prep job ${PREP_JID} (noise on canonical HF images)"

    # ── 2. Compute jobs: one per (model x level) ──
    JOBIDS=()
    for MODEL in "${MODELS[@]}"; do
        WALL=$(walltime_for "$MODEL")
        for LEVEL in "${LEVELS[@]}"; do
            OUT=$(sbatch ${PREP_DEP} <<SCRIPT
#!/bin/bash -l
#SBATCH --job-name=lg-${BM}-${MODEL:0:8}-L${LEVEL}
#SBATCH -p ${PARTITION}
#SBATCH -w ${NODE}
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${WALL}
#SBATCH --mail-user=rg21@usf.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/legib_${BM}_${MODEL}_L${LEVEL}_%j.log
#SBATCH --error=logs/legib_${BM}_${MODEL}_L${LEVEL}_%j.err

conda activate vlm
export HF_HOME="${HF_HOME}"
cd ${REPO_DIR}
echo "=== ${BM} | ${MODEL} | level ${LEVEL} on \$(hostname) : \$(date) ==="
srun python scripts/run_legibility.py --benchmark ${BM} --models ${MODEL} --noise-levels ${LEVEL} --num-problems ${NUM_PROBLEMS} --output-dir "${OUTPUT_DIR}"
SCRIPT
)
            JID=$(echo "$OUT" | awk '{print $NF}')
            JOBIDS+=("$JID")
            TOTAL=$((TOTAL+1))
            echo "  submitted ${BM} ${MODEL} L${LEVEL} (job ${JID}, wall ${WALL})"
        done
    done

    # ── 3. Merge job for this benchmark (after all its cells succeed) ──
    DEP=$(IFS=:; echo "${JOBIDS[*]}")
    MERGE_MODELS="${MODELS[*]}"
    sbatch --dependency=afterok:${DEP} <<SCRIPT >/dev/null
#!/bin/bash -l
#SBATCH --job-name=legib-merge-${BM}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/legib_merge_${BM}_%j.log
#SBATCH --error=logs/legib_merge_${BM}_%j.err

conda activate vlm
cd ${REPO_DIR}
srun python scripts/run_legibility.py --merge --benchmark ${BM} --models ${MERGE_MODELS} --num-problems ${NUM_PROBLEMS} --output-dir "${OUTPUT_DIR}"
SCRIPT
    echo "[${BM}] merge job submitted (depends on ${#JOBIDS[@]} cells)"
done

echo ""
echo "Submitted ${TOTAL} compute cells across ${#BENCHMARKS[@]} benchmarks + prep/merge jobs."
echo "Monitor with: squeue -u \$USER"
echo "Curves per benchmark:"
for BM in "${BENCHMARKS[@]}"; do
    if [ "$BM" = "gsm8k" ]; then echo "  ${OUTPUT_DIR}/legibility_all.json";
    else echo "  ${OUTPUT_DIR}/${BM}/legibility_all.json"; fi
done
