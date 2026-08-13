#!/bin/bash
#SBATCH --job-name=snn_fullscale_rho
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# Full-scale ranking-vs-ground-truth correlation: diagnose_fullscale_rho.py.
#
# One GPU, well under an hour at the default 64 channels per layer: the cost
# is (channels sampled) x (val-subset forwards), no training. Arguments pass
# straight through, and are required -- the script itself refuses to run
# without --model and --tag:
#
#   sbatch slurm_rho.sh --model spear_repl_resnet18 --tag lagrw
#   sbatch slurm_rho.sh --model spear_repl --tag lagrw
# ---------------------------------------------------------------------------

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "ERROR: no arguments given. Needs at least --model and --tag, e.g." >&2
    echo "  sbatch slurm_rho.sh --model spear_repl_resnet18 --tag lagrw" >&2
    exit 2
fi

echo "===================================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

VENV_PATH="./venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH ..."
    python -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

mkdir -p logs

python diagnose_fullscale_rho.py "$@"

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
