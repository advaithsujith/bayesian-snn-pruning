#!/bin/bash
#SBATCH --job-name=bayesian_snn_pruning
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# SLURM launch script for the full Bayesian SNN pruning pipeline.
#
# Tuned for University of Manchester CSF3
# (https://ri.itservices.manchester.ac.uk/csf3/batch-slurm/partitions/):
#   - gpuA: 4x Nvidia A100 80GB, 4-day (96h) max wallclock, general access.
#   - gpuL: 4x Nvidia L40S 48GB, 4-day max, general access -- swap
#     --partition=gpuA for --partition=gpuL below if gpuA is busy/queued.
#   - CSF3 requests GPUs via `-G NUM`, not the generic `--gres=gpu:N`
#     syntax some other clusters use.
#   - gpuA40GB and gpuH/gpuH_short require separate restricted-access
#     approval and are not used here.
#
# If you're on a different cluster, adjust --partition, the GPU request
# line, --cpus-per-task, --mem and --time to match its queue names and
# resource limits before submitting (`sbatch slurm.sh`).
#
# 96h may not be enough for all three architectures at full epoch counts
# on a single job -- if a sanity-check run (see README) shows per-epoch
# time is too slow to fit, split this into three separate sbatch
# submissions (one per architecture) instead of one combined job.
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
# Adjust this path, or replace with `conda activate <env_name>` if your
# cluster manages environments via conda instead of a venv.
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

# --- Run the full pipeline --------------------------------------------------
mkdir -p checkpoints outputs/lenet outputs/vgg9 outputs/resnet18 plots logs

python run_all.py

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
