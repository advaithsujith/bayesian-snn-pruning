#!/bin/bash
#SBATCH --job-name=snn_sparsity_curve
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
# Arguments are required, and are passed straight through to
# run_sparsity_curve.py, so each experimental arm is one submission line:
#
#   sbatch slurm_curve.sh --model dpap_repl --tag sgd \
#       --gate-optimizer sgd --gate-lr 0.05 --targets 20 33.46 50.80 70 90
#
# The wallclock above is a request, not a reservation, and SLURM will not
# backfill a job into a gap shorter than what it asks for. This previously
# requested 96h for runs that finish in one to six, which is how a 32-minute
# gate test ended up with a start time 14 hours after submission. Override it
# per arm rather than raising the default: `sbatch --time=01:00:00` for a gate
# test, the default for a full five-point curve.
#
# Before submitting: python tests/test_ranked_pruning.py &&
#   python tests/test_vggstyle.py && python tests/test_synops.py
# ---------------------------------------------------------------------------

set -euo pipefail

# Checked before the venv build and the pip install, so a mis-submission
# releases the GPU allocation in a second rather than after two minutes of
# environment setup.
#
# A bare `sbatch slurm_curve.sh` used to run a default arm: dpap_repl at
# --tag beta0.01, over the five standard targets. That default is gone, and
# deliberately so.
#
# run_sparsity_curve.py writes into sparsity_curve_<tag> without clearing it,
# so the tag is the only thing separating one run's evidence from another's.
# beta0.01 is the archived run behind the +1.54 to +5.25pp margin over the
# random control, and its bayesian_model.pt is gitignored like every other
# weight file -- so the no-argument path pointed at the one directory in this
# repo holding an irreplaceable artifact. Submitted bare by accident on
# 2026-08-13; caught 8 minutes in, before gate training finished and reached
# the torch.save. Nothing was lost, and nothing about the setup made that
# outcome likely rather than lucky.
#
# Requiring an explicit --tag costs one line per submission and removes the
# failure mode. The gate-pressure diagnostic that used to run as a pre-flight
# here is worth keeping, but as its own deliberate submission, since it reads
# a ratio a human then has to act on:
#
#   sbatch --time=00:20:00 slurm_curve.sh --model spear_repl_resnet18 \
#       --diagnose-only --fail-below 1e-3 --fail-above 2e-2
if [ "$#" -eq 0 ]; then
    echo "ERROR: no arguments given." >&2
    echo >&2
    echo "This script has no default arm. Every run needs at least --model" >&2
    echo "and --tag, because the tag is what stops one arm overwriting" >&2
    echo "another's results. Examples:" >&2
    echo >&2
    echo "  sbatch --time=01:00:00 slurm_curve.sh --model spear_repl_resnet18 \\" >&2
    echo "      --tag sgd2 --gate-optimizer sgd --gate-lr 0.01 --beta-max 0.05 \\" >&2
    echo "      --targets 33.46 50.80 --finetune-epochs 30" >&2
    echo >&2
    echo "  sbatch slurm_curve.sh --model dpap_repl --tag beta0.01_seed1 \\" >&2
    echo "      --seed 1 --targets 20 33.46 50.80 70 90" >&2
    echo >&2
    echo "See HANDOFF.md for which arms are outstanding." >&2
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

nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir -p checkpoints outputs plots logs

# Note run_sparsity_curve.py stops before the fine-tunes if the gates did not
# differentiate, so a bad gate phase still costs ~1.5h rather than ~4h.
python run_sparsity_curve.py "$@"

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
