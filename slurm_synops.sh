#!/bin/bash
#SBATCH --job-name=baseline_synops
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/synops_%j.out
#SBATCH --error=logs/synops_%j.err

# ---------------------------------------------------------------------------
# Measure the unpruned baseline SynOps.
#
# No training and no gradients: this loads the saved baseline checkpoint and
# runs a handful of forward passes with hooks attached. Minutes at most, hence
# the 20-minute wallclock rather than slurm.sh's 96 hours. Swap
# --partition=gpuA for --partition=gpuL if gpuA is queued.
# ---------------------------------------------------------------------------

set -euo pipefail

module purge
module load libs/cuda

if [ ! -d venv ]; then
    echo "No venv found, creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

mkdir -p logs

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python measure_baseline_synops.py --model dpap_repl --batches 8

echo "Done. Result written to outputs/dpap_repl/baseline_synops.json"
