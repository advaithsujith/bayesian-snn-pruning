#!/bin/bash
#SBATCH --job-name=snn_gate_toy
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# CPU-only diagnostic study: diagnose_gate_toy.py.
#
# Deliberately requests no GPU and no partition: the study is a miniature of
# the gate-training pipeline on 8x8 synthetic data, sized for minutes of CPU,
# and asking for a gpuA slot would queue it behind real training runs for no
# benefit. If this cluster rejects partitionless submissions, add the CPU
# partition explicitly: `sbatch -p multicore slurm_toy.sh`.
#
# What it answers, and why it is worth a job at all: whether the gate
# machinery recovers a ground-truth (ablation-measured) channel ranking
# under dpap-like versus SPEAR-like SNN settings, crossed with the working
# and failing optimizer mechanics, then flipping the SPEAR ingredients one
# at a time. See diagnose_gate_toy.py's docstring for the cell layout and
# HANDOFF.md for the two NOT USABLE runs that motivated it.
# ---------------------------------------------------------------------------

set -euo pipefail

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python diagnose_gate_toy.py

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
