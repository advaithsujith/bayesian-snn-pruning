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
