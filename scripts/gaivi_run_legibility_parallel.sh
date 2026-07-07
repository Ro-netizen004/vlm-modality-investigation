#!/bin/bash -l
# ── Legibility experiment, fanned out across the cluster ──
# Submits ONE SLURM job per (model, noise-level) so they run concurrently on the
# 8x L40S in GPU53, instead of one serial job on a single GPU (which hit the 12h
# wall-time after ~4 levels of a single model).
#
# Grid: 2 anchor models x 4 monotonic legibility levels = 8 jobs -> one wave on
# the 8 L40S -> wall clock ~= the slowest single (model, level) cell (~3h for
# Idefics3), not 24h+ serial.
#
# Each cell is self-contained and checkpoints per problem (run_legibility.py),
# so a cell killed at its wall-time resumes on resubmission instead of restarting.
# A final merge job (SLURM dependency) collates per-level JSONs into the summaries.
#
# Usage:
#   cd ~/vlm-modality-research
#   bash scripts/gaivi_run_legibility_parallel.sh
#
# Monitor:  squeue -u $USER
# Cancel:   scancel -u $USER
# Merge by hand (if the dep job didn't run):
#   python scripts/run_legibility.py --merge --models "${MODELS[@]}" \
#          --output-dir "$OUTPUT_DIR" --num-problems "$NUM_PROBLEMS"

PARTITION="CISL"
NODE="GPU53"                                   # 8x L40S
REPO_DIR="$HOME/vlm-modality-research"         # <-- match your checkout dir
OUTPUT_DIR="$HOME/vlm_research_results/phase6_legibility"
NOISE_IMAGES="$HOME/vlm_research_results/phase4/images"
export HF_HOME="${HF_HOME:-/data/rg21/hf_cache}"

# NUM_PROBLEMS=50 matches the existing lean run. Now that cells run in parallel
# you can afford more decidable trials — bump to 100-150 for a firmer text-pref
# estimate, but FIRST clear stale level_*.json (they're keyed by level, not N,
# so old N=50 files would be wrongly reused). See --merge note above.
NUM_PROBLEMS=50

MODELS=("Idefics3-8B-Llama3" "Qwen2.5-VL-7B-Instruct")   # vulnerable + resilient
LEVELS=(0 2 4 5)   # monotonic legibility ladder: clean -> light blur -> blur+noise -> heavy
                   # (skip 6-9: rotation/handwriting/screenshot/combined confound legibility)

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs" "$HF_HOME"

# Idefics3 is ~3h/level (heavy visual-token splitting); Qwen2.5-VL-7B is far
# faster. Give the slow model generous wall-time, the fast one a short slot.
walltime_for() {
    case "$1" in
        Idefics3-8B-Llama3) echo "05:00:00" ;;
        *)                  echo "02:00:00" ;;
    esac
}

echo "Submitting $(( ${#MODELS[@]} * ${#LEVELS[@]} )) (model x level) jobs to ${NODE}..."
echo ""

JOBIDS=()
for MODEL in "${MODELS[@]}"; do
    WALL=$(walltime_for "$MODEL")
    for LEVEL in "${LEVELS[@]}"; do
        OUT=$(sbatch <<SCRIPT
#!/bin/bash -l
#SBATCH --job-name=legib-${MODEL:0:10}-L${LEVEL}
#SBATCH -p ${PARTITION}
#SBATCH -w ${NODE}
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${WALL}
#SBATCH --mail-user=rg21@usf.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/legib_${MODEL}_L${LEVEL}_%j.log
#SBATCH --error=logs/legib_${MODEL}_L${LEVEL}_%j.err

conda activate vlm
export HF_HOME="${HF_HOME}"
cd ${REPO_DIR}

echo "=== ${MODEL}  level ${LEVEL}  on \$(hostname) : \$(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

srun python scripts/run_legibility.py \\
    --models ${MODEL} \\
    --noise-levels ${LEVEL} \\
    --num-problems ${NUM_PROBLEMS} \\
    --noise-image-dir "${NOISE_IMAGES}" \\
    --output-dir "${OUTPUT_DIR}"

echo "=== done ${MODEL} L${LEVEL} : \$(date) ==="
SCRIPT
)
        JID=$(echo "$OUT" | awk '{print $NF}')
        JOBIDS+=("$JID")
        echo "  Submitted: ${MODEL} L${LEVEL}  (job ${JID}, wall ${WALL})"
    done
done

# ── Final merge job: runs only after every cell succeeds ──
DEP=$(IFS=:; echo "${JOBIDS[*]}")
MERGE_MODELS="${MODELS[*]}"
sbatch --dependency=afterok:${DEP} <<SCRIPT >/dev/null
#!/bin/bash -l
#SBATCH --job-name=legib-merge
#SBATCH -p ${PARTITION}
#SBATCH -w ${NODE}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --mail-user=rg21@usf.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/legib_merge_%j.log
#SBATCH --error=logs/legib_merge_%j.err

conda activate vlm
cd ${REPO_DIR}
srun python scripts/run_legibility.py --merge \\
    --models ${MERGE_MODELS} \\
    --num-problems ${NUM_PROBLEMS} \\
    --output-dir "${OUTPUT_DIR}"
SCRIPT

echo ""
echo "All ${#JOBIDS[@]} cell jobs + 1 merge job submitted."
echo "Monitor with: squeue -u \$USER"
echo "Final curves: ${OUTPUT_DIR}/legibility_all.json"
