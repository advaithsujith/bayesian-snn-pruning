#!/bin/bash
#SBATCH --job-name=snn_beta_sweep
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# SUPERSEDED -- submit slurm_curve.sh instead.
#
# This sweeps beta_max in the hope of landing near a target sparsity. Under
# Adam that does not work as intended: beta_max sets where the task and KL
# gradients cancel, not how fast log_alpha moves, so a spread of values
# gives a cliff rather than a curve -- at roughly 2h per point. And with
# dpap_repl's val_fraction=0.0 it was ranking candidates by test accuracy.
# See sweep_beta.py's module docstring for the full argument.
#
# run_sparsity_curve.py reaches any sparsity from a single gate-training run,
# so there is nothing left for this to select.
#
# beta_max sweep against a frozen pretrained baseline.
#
# Each trial runs only the stages that depend on beta_max (gate training,
# pruning, fine-tuning, evaluation) and reuses one saved baseline, so a trial
# costs roughly gate-training + fine-tuning rather than a full pipeline.
# Requires outputs/<model>/trained_model.pt to already exist -- run slurm.sh
# once first if it does not.
#
# See slurm.sh for the CSF3 partition/GPU-request notes.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "slurm_sweep.sh is superseded by slurm_curve.sh -- see the header." >&2
echo "Remove this guard deliberately if you really do want the sweep." >&2
exit 1

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

nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir -p checkpoints outputs plots logs

# Ordered high -> low. 0.4 is the value already known to collapse this
# config; it is included as an in-sweep control so the trend is visible
# rather than inferred from a previous run.
python sweep_beta.py --model dpap_repl --betas 0.4 0.05 0.02 0.005

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
