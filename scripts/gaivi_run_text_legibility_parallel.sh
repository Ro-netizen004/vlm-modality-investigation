#!/bin/bash -l
# ── Mirror arm (Phase 7): TEXT-degradation legibility, fanned out on the cluster ──
# Symmetric counterpart to gaivi_run_legibility_parallel.sh. That script degrades
# the IMAGE and holds the text clean (Phase 6); this one holds the IMAGE clean and
# degrades the TEXT channel (src/text_noise.py), to test whether modality preference
# tracks TEXT reliability the way it tracks image reliability.
#
# Key differences from the image arm:
#   * Runs with --channel text, so results land under text_legibility/ and the
#     text is corrupted per src/text_noise.py (levels 0/2/4/5) instead of the image.
#   * Text degradation touches only the STRING, so every level reads the SAME
#     clean level-0 image. There is no per-level image cost, so we run ALL levels
#     of a model in ONE job (one job per model) rather than one job per (model,level).
#   * The prep job renders only the level-0 clean images — that's all the text arm
#     needs (run_legibility.py auto-includes level 0 for the text channel anyway).
#
# Grid: 2 benchmarks x 8 models = 16 compute jobs (each running levels 0 2 4 5).
#
# Usage:
#   cd ~/vlm-modality-research
#   bash scripts/gaivi_run_text_legibility_parallel.sh
#
# Monitor:  squeue -u $USER      Cancel: scancel -u $USER

PARTITION="CISL"
NODE="GPU53"                                   # 8x L40S
REPO_DIR="$HOME/vlm-modality-investigation"    # <-- your GAIVI checkout dir
OUTPUT_DIR="$HOME/vlm_research_results/phase6_legibility"
export HF_HOME="${HF_HOME:-/data/rg21/hf_cache}"

# Paper-scale default: full SVAMP (300); solid GSM8K subset. run_legibility.py
# auto-invalidates stale level_*.json when --num-problems changes.
NUM_PROBLEMS="${NUM_PROBLEMS:-300}"

# Grid is overridable via env vars for targeted runs (defaults = full headline grid).
IFS=' ' read -ra BENCHMARKS <<< "${BENCHMARKS_OVERRIDE:-gsm8k svamp}"   # numeric Protocol-A benchmarks
IFS=' ' read -ra MODELS <<< "${MODELS_OVERRIDE:-Qwen2-VL-2B-Instruct llava-v1.6-mistral-7b-hf Qwen2.5-VL-7B-Instruct Idefics3-8B-Llama3 MiniCPM-V-2_6 InternVL2-8B llava-onevision-qwen2-7b-ov-hf Phi-3.5-vision-instruct}"
# Text-corruption ladder mirrors the image ladder's clean -> heavy intent.
LEVELS_STR="${LEVELS_OVERRIDE:-0 2 4 5}"

# Also compute the CLL arbitration-margin curve (true mirror of the image-arm CLL).
# On by default: it runs in the SAME job, right after the generation pass, so it can
# join the reasoning label from the generation CSVs. Non-CLL model types self-skip.
# Set RUN_CLL=0 to run generation only (halves wall-time).
RUN_CLL="${RUN_CLL:-1}"

mkdir -p "$OUTPUT_DIR" "$REPO_DIR/logs" "$HF_HOME"

# All levels run in a single job per model, so give each the wall-time it would
# have needed for the FULL image-arm level sweep (4 levels back-to-back). When the
# CLL pass is enabled it runs in the same job (two teacher-forced scorings per trial
# on top of generation), so budget extra headroom.
walltime_for() {
    if [ "$RUN_CLL" = "1" ]; then
        case "$1" in
            Idefics3-8B-Llama3)                                echo "24:00:00" ;;
            MiniCPM-V-2_6|llava-onevision-qwen2-7b-ov-hf|\
            llava-v1.6-mistral-7b-hf|Phi-3.5-vision-instruct)  echo "16:00:00" ;;
            *)                                                 echo "10:00:00" ;;
        esac
    else
        case "$1" in
            Idefics3-8B-Llama3)                                echo "16:00:00" ;;
            MiniCPM-V-2_6|llava-onevision-qwen2-7b-ov-hf|\
            llava-v1.6-mistral-7b-hf|Phi-3.5-vision-instruct)  echo "10:00:00" ;;
            *)                                                 echo "06:00:00" ;;
        esac
    fi
}

TOTAL=0
for BM in "${BENCHMARKS[@]}"; do

    # ── 1. Prep job: render the level-0 clean canonical images once (all the text arm needs) ──
    PREP_OUT=$(sbatch <<SCRIPT
#!/bin/bash -l
#SBATCH --job-name=txtlegib-prep-${BM}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/txtlegib_prep_${BM}_%j.log
#SBATCH --error=logs/txtlegib_prep_${BM}_%j.err

conda activate vlm
export HF_HOME="${HF_HOME}"
cd ${REPO_DIR}
srun python scripts/run_legibility.py --render-only --channel text --benchmark ${BM} --num-problems ${NUM_PROBLEMS} --noise-levels 0 --output-dir "${OUTPUT_DIR}"
SCRIPT
)
    PREP_JID=$(echo "$PREP_OUT" | awk '{print $NF}')
    PREP_DEP="--dependency=afterok:${PREP_JID}"
    echo "[${BM}] prep job ${PREP_JID} (level-0 clean canonical images for text arm)"

    # ── 2. Compute jobs: one per model, all text levels in the same job ──
    JOBIDS=()
    for MODEL in "${MODELS[@]}"; do
        WALL=$(walltime_for "$MODEL")
        # Optional CLL pass (true mirror of the image-arm CLL), same job, after generation.
        if [ "$RUN_CLL" = "1" ]; then
            CLL_STEP="srun python scripts/run_legibility.py --score-cll --channel text --benchmark ${BM} --models ${MODEL} --noise-levels ${LEVELS_STR} --num-problems ${NUM_PROBLEMS} --output-dir \"${OUTPUT_DIR}\""
        else
            CLL_STEP=":"   # no-op
        fi
        OUT=$(sbatch ${PREP_DEP} <<SCRIPT
#!/bin/bash -l
#SBATCH --job-name=txlg-${BM}-${MODEL:0:8}
#SBATCH -p ${PARTITION}
#SBATCH -w ${NODE}
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${WALL}
#SBATCH --mail-user=rg21@usf.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/txtlegib_${BM}_${MODEL}_%j.log
#SBATCH --error=logs/txtlegib_${BM}_${MODEL}_%j.err

conda activate vlm
export HF_HOME="${HF_HOME}"
cd ${REPO_DIR}
echo "=== ${BM} | ${MODEL} | text channel, levels ${LEVELS_STR} on \$(hostname) : \$(date) ==="
srun python scripts/run_legibility.py --channel text --benchmark ${BM} --models ${MODEL} --noise-levels ${LEVELS_STR} --num-problems ${NUM_PROBLEMS} --output-dir "${OUTPUT_DIR}"
${CLL_STEP}
SCRIPT
)
        JID=$(echo "$OUT" | awk '{print $NF}')
        JOBIDS+=("$JID")
        TOTAL=$((TOTAL+1))
        echo "  submitted ${BM} ${MODEL} text-arm (job ${JID}, wall ${WALL})"
    done

    # ── 3. Merge job for this benchmark (after all its cells succeed) ──
    DEP=$(IFS=:; echo "${JOBIDS[*]}")
    MERGE_MODELS="${MODELS[*]}"
    sbatch --dependency=afterok:${DEP} <<SCRIPT >/dev/null
#!/bin/bash -l
#SBATCH --job-name=txtlegib-merge-${BM}
#SBATCH -p ${PARTITION}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/txtlegib_merge_${BM}_%j.log
#SBATCH --error=logs/txtlegib_merge_${BM}_%j.err

conda activate vlm
cd ${REPO_DIR}
srun python scripts/run_legibility.py --merge --channel text --benchmark ${BM} --models ${MERGE_MODELS} --num-problems ${NUM_PROBLEMS} --output-dir "${OUTPUT_DIR}"
SCRIPT
    echo "[${BM}] merge job submitted (depends on ${#JOBIDS[@]} cells)"
done

echo ""
echo "Submitted ${TOTAL} text-arm compute cells across ${#BENCHMARKS[@]} benchmarks + prep/merge jobs."
echo "Monitor with: squeue -u \$USER"
echo "Curves per benchmark (text arm):"
for BM in "${BENCHMARKS[@]}"; do
    if [ "$BM" = "gsm8k" ]; then echo "  ${OUTPUT_DIR}/text_legibility/legibility_all.json";
    else echo "  ${OUTPUT_DIR}/${BM}/text_legibility/legibility_all.json"; fi
done
