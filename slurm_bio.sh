#!/bin/bash
#SBATCH --job-name=bio_snn_pruning
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm_bio_%j.out
#SBATCH --error=logs/slurm_bio_%j.err

# ---------------------------------------------------------------------------
# SLURM launch script for the bio-inspired pruning baselines
# (activity_pruning.py / run_bio_pruning.py). Same cluster tuning as
# slurm.sh -- see that file's header for the CSF3 partition notes.
#
# Run this AFTER slurm.sh has produced outputs/<model>/trained_model.pt
# for the architectures you care about, so the bio-inspired criteria fork
# from the exact same pretrained weights the Bayesian pipeline used (if a
# given architecture's trained_model.pt is missing, run_bio_pruning.py
# will just train and cache its own -- see README's "Bio-inspired pruning
# baselines" section for the run-order caveat this implies).
#
# 96h may not be enough for all three architectures x three criteria x
# every keep_fraction in config.py's BioPruningConfig at full epoch
# counts in one job -- if a sanity-check run shows per-epoch time is too
# slow to fit, split into separate sbatch submissions.
#
# Arguments given to sbatch pass straight through to run_bio_pruning.py, so
# splitting the sweep no longer means editing the script. The
# matched-sparsity comparison against the Bayesian curve is one line:
#
#   sbatch slurm_bio.sh --models dpap_repl --keep-fractions 0.82 0.70 \
#       --output outputs/bio_results_dpap_repl.csv
#
# (take the keep-fractions from
#  `python run_sparsity_curve.py --model dpap_repl --targets ... --plan-only`,
#  which converts a parameter-pruning target into the per-layer keep
#  fraction the bio criteria consume -- the two are not the same number.)
# ---------------------------------------------------------------------------

set -euo pipefail

echo "===================================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

# --- Environment modules -----------------------------------------------
# Uncomment / edit for your cluster's module system.
# module purge
# module load cuda/12.1
# module load python/3.10
# module load anaconda3

# --- Python environment --------------------------------------------------
VENV_PATH="./venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH ..."
    python -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# --- Hardware info ---------------------------------------------------------
echo "----------------------------------------------------"
echo "Hardware information"
echo "----------------------------------------------------"
nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "----------------------------------------------------"

# --- Run the bio-inspired pruning pipeline ----------------------------------
mkdir -p checkpoints outputs/lenet outputs/vgg9 outputs/resnet18 plots logs

python run_bio_pruning.py "$@"

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
