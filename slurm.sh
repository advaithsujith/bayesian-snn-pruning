#!/bin/bash
#SBATCH --job-name=bayesian_snn_pruning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# SLURM launch script for the full Bayesian SNN pruning pipeline.
#
# Adjust --partition, --gres, --cpus-per-task, --mem and --time to match
# your specific university cluster's queue names and resource limits
# before submitting (`sbatch slurm.sh`).
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
