#!/bin/bash
#SBATCH --job-name=snn_sparsity_curve
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# Accuracy-vs-sparsity curve from ONE gate-training run.
#
# Replaces slurm_sweep.sh. That script re-ran gate training per beta_max
# value in the hope of landing near a sparsity; this one trains gates once
# and then rebuilds at each requested sparsity directly, so a point costs a
# fine-tune (~25 min here) instead of gate training plus a fine-tune (~2h),
# and the sparsity is an input rather than an outcome.
#
# The two targets below are DPAP's own published operating points (94.27% at
# 33.46% pruned, 93.83% at 50.80%), so the comparison is against their
# numbers at their sparsity rather than at whichever sparsity a threshold
# happened to produce. The extra points fill in the curve either side.
#
# Requires outputs/dpap_repl/trained_model.pt (the 94.35% replicated
# baseline). See slurm.sh for the CSF3 partition/GPU-request notes.
#
# Any arguments given to sbatch are passed straight through to
# run_sparsity_curve.py, so each experimental arm is one submission line:
#
#   sbatch slurm_curve.sh --model dpap_repl --tag sgd \
#       --gate-optimizer sgd --gate-lr 0.05 --targets 20 33.46 50.80 70 90
#
# With no arguments it runs the default beta0.01 Adam curve below.
#
# Before submitting: python tests/test_ranked_pruning.py &&
#   python tests/test_vggstyle.py && python tests/test_synops.py
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

# Tagged so this does not overwrite the previous run. The 2026-08-03 run at
# beta_max=0.4 finished with 89% of gates pinned at the clamp, so its curve
# measures random structured pruning rather than the criterion; keep it as
# outputs/dpap_repl/sparsity_curve_RANDOM_CONTROL, it is the bar this run
# has to beat (84.14% at 90% pruned, 90.62% at 50.80%).
#
# Stops before the five fine-tunes if the gates did not differentiate, so a
# bad gate phase costs ~1.5h rather than ~4h.
if [ "$#" -gt 0 ]; then
    python run_sparsity_curve.py "$@"
else
    # Pre-flight: can the task loss push back against the KL at all? One
    # batch, under a minute. If the weakest layer's ratio is ~1e-3 or below,
    # the KL is unopposed there, gates will saturate, and every point on the
    # curve below would be ties broken by index order. Cheaper to find out
    # here. Only run for the default invocation -- a passthrough arm may
    # target a different model, and under set -e a wrong-model diagnosis
    # would kill the job.
    python run_sparsity_curve.py --model dpap_repl --diagnose-only

    python run_sparsity_curve.py \
        --model dpap_repl \
        --mode uniform \
        --tag beta0.01 \
        --targets 20 33.46 50.80 70 90
fi

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
