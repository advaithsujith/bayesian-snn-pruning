#!/bin/bash
#SBATCH --job-name=hpo_snn_pruning
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm_hpo_%j.out
#SBATCH --error=logs/slurm_hpo_%j.err

# ---------------------------------------------------------------------------
# SLURM launch script for the multi-sampler hyperparameter search
# (hpo_search.py). Same cluster tuning as slurm.sh -- see that file's header
# for the CSF3 partition notes.
#
# Requires outputs/lenet/trained_model.pt to already exist (from a prior
# slurm.sh or slurm_bio.sh run) so the search forks from the same pretrained
# checkpoint everything else does; if missing, hpo_search.py trains and
# caches its own, same self-healing behaviour as run_bio_pruning.py.
#
# Estimated runtime: ~45-60h for the search phase (135 shortened trials,
# LeNet-only) plus ~6h for the three full-length confirmation runs -- should
# fit CSF3's 96h cap, but it's an estimate, not a measurement. If a
# sanity-check run shows per-trial timing is slower than expected, split by
# method across multiple submissions instead of one combined job, e.g.:
#   sbatch --wrap="python hpo_search.py --methods bayesian"
#   sbatch --wrap="python hpo_search.py --methods sca"
#   sbatch --wrap="python hpo_search.py --methods dpap"
# (each writes its own outputs/hpo/<method>_*.csv independently, so partial
# runs don't clobber each other -- just rerun main() once all three methods'
# CSVs exist if you want the combined confirmation-run step, or run the
# confirmation runs by hand from the saved per-sampler best hyperparameters.)
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

# --- Run the hyperparameter search ------------------------------------------
mkdir -p checkpoints outputs/lenet outputs/hpo plots logs

python hpo_search.py

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
