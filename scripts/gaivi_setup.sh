#!/bin/bash -l
# ── One-time setup for GAIVI ──
# Run this ONCE after SSH'ing into GAIVI (on the login node, not in a job).
#
# Usage:
#   ssh aviralgupta@gaivi.cse.usf.edu
#   bash scripts/gaivi_setup.sh
#
# This creates the conda environment and clones the repo.

set -euo pipefail

echo "============================================"
echo "  GAIVI Setup — VLM Modality Research"
echo "============================================"

# ── Step 1: Create conda environment ──
# GAIVI requires you to create your own env (can't install to base)
echo ""
echo "Step 1: Creating conda environment 'vlm'..."

conda create --name vlm python=3.10 -y
conda activate vlm

# ── Step 2: Install PyTorch with CUDA ──
echo ""
echo "Step 2: Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── Step 3: Install project dependencies ──
echo ""
echo "Step 3: Installing project dependencies..."
pip install transformers>=4.45.0 accelerate datasets evaluate scipy numpy pandas matplotlib Pillow tqdm pyyaml bitsandbytes pyarrow huggingface_hub

# ── Step 4: Clone the repo (if not already) ──
echo ""
echo "Step 4: Setting up repo..."
REPO_DIR="$HOME/vlm-modality-investigation"
if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists at $REPO_DIR — pulling latest..."
    cd "$REPO_DIR" && git pull origin main
else
    echo "Cloning repo..."
    git clone https://github.com/Ro-netizen004/vlm-modality-investigation.git "$REPO_DIR"
fi

# ── Step 5: Create output directory ──
OUTPUT_DIR="$HOME/vlm_research_results"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO_DIR/logs"

# ── Step 6: Check partition access ──
echo ""
echo "Step 5: Your available partitions:"
sinfo
echo ""
echo "Look for a CISL or Contributors partition that includes gpu53."
echo "Update the -p flag in the SLURM scripts if needed."

# ── Done ──
echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Conda env:  vlm"
echo "  Repo:       $REPO_DIR"
echo "  Output:     $OUTPUT_DIR"
echo ""
echo "  Next steps:"
echo "    1. Check which partition has gpu53 (from sinfo output above)"
echo "    2. Edit scripts/gaivi_run_all.sh if partition isn't 'CISL'"
echo "    3. Run: cd $REPO_DIR && sbatch scripts/gaivi_run_all.sh"
echo "============================================"
