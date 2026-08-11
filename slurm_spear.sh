#!/bin/bash
#SBATCH --job-name=spear_repl_pretrain
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spear_%j.out
#SBATCH --error=logs/spear_%j.err

# ---------------------------------------------------------------------------
# SPEAR replication, stage 1: pretrain the VGG16 baseline.
#
# Replicates the CIFAR-10 / VGG16 setup of Xie et al., "SPEAR: Structured
# Pruning for Spiking Neural Networks via Synaptic Operation Estimation and
# Reinforcement Learning" (arXiv 2507.02945). Setup provenance is recorded
# row-by-row in docs/replication_targets.md section 4; the config is
# config.get_spear_repl_config().
#
# Recipe: VGG16, T=4, LIF (hard reset, threshold 1.0, tau 2.0, arctan
# surrogate), TET loss, SGD momentum 0.9, wd 5e-5, max lr 0.1, 210 epochs
# (10 linear warm-up + 200 cosine), no augmentation.
#
# WHY --pretrain-only: gate training is deliberately NOT run here. beta_max
# does not transfer between setups -- it balances the KL against the task
# loss's gradient on log_alpha, and that ratio moves with architecture,
# timestep count, and the analog output readout this config introduces.
# Borrowing a value on a plausibility argument is what collapsed three DPAP
# runs. Read the gate-pressure diagnostic on THIS baseline first.
#
# Wallclock: 24h, against an estimated ~4-6h for 210 epochs. Generous
# because per-epoch time for VGG16 at T=4 has never been measured on CSF3 --
# tighten it once one run has reported.
#
# Swap --partition=gpuA for --partition=gpuL if gpuA is queued.
#
# BEFORE SUBMITTING:
#   git pull
#   source venv/bin/activate
#   python tests/test_spear.py
#   python tests/test_vggstyle.py
#   python tests/test_ranked_pruning.py
#   python tests/test_synops.py
# The tests fail with ModuleNotFoundError outside the venv while sbatch
# proceeds regardless; that has happened once already.
#
# CIFAR-10 must already be downloaded -- GPU compute nodes have no internet.
# Pre-download on the login node if ./data is empty.
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

mkdir -p checkpoints outputs/spear_repl plots logs

echo "===================================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Fail fast rather than 200 epochs deep, and on the login node's schedule
# rather than the queue's.
python tests/test_spear.py

python run_all.py --model spear_repl --pretrain-only

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
echo ""
echo "Baseline saved to outputs/spear_repl/trained_model.pt"
echo ""
echo "NOTE: SPEAR publishes no unpruned baseline accuracy, so there is no"
echo "go/no-go gate here as there was for DPAP's 94.54%. The nearest"
echo "published reference is SCA's 91.14% (SPEAR quotes SCA's pruned rows"
echo "verbatim, and both use VGG16 at T=4). Landing far below that means"
echo "diagnose before pruning; landing near it is the expected outcome."
echo ""
echo "Next:"
echo "  1. push outputs/spear_repl off scratch"
echo "  2. sbatch slurm_synops.sh   (edit --model to spear_repl)"
echo "  3. set reuse_pretrained=True in get_spear_repl_config()"
echo "  4. measure gate pressure before trusting beta_max=0.01"
