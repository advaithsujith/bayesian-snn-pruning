#!/bin/bash
#SBATCH --job-name=snn_compare
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

# ---------------------------------------------------------------------------
# Full five-criterion comparison for ONE platform, in one submission.
#
#   sbatch slurm_compare.sh spear_repl
#   sbatch slurm_compare.sh spear_repl_resnet18
#   sbatch slurm_compare.sh dpap_repl
#
# Optional 2nd/3rd args: finetune epochs (default 30) and the parameter
# targets (default "33.46 50.80 70 90").
#
#   sbatch slurm_compare.sh spear_repl 30 "20 33.46 50.80 70 90"
#
# Requires outputs/<model>/trained_model.pt to exist already -- run the
# relevant pretrain job first (slurm_spear.sh / slurm_spear_resnet18.sh), and
# set reuse_pretrained=True in that config.
#
# WHY ONE PLATFORM PER JOB, not all three: a 10h+ job that dies at hour 8
# loses everything after the last checkpoint, queues longer for a big slot,
# and gives you no chance to look at the gate diagnostic before the rest
# proceeds. Three independent ~3-4h jobs are individually resubmittable.
#
# Stages:
#   1. gate-pressure diagnostic, and ABORT if the KL is unopposed. One batch.
#      This is the guard that dpap_repl did not have: its diagnostic printed
#      1.4e-4 before epoch 1, was ignored, and the run lost all accuracy by
#      epoch 15.
#   2. Bayesian accuracy-vs-sparsity curve (also stops itself if the gates
#      failed to differentiate -- see KeepPlan.ranking_is_usable).
#   3. derive the matching keep_fractions from the same geometry the Bayesian
#      plan used, and pass them straight through. No hand transcription: doing
#      that by eye is what previously compared a 27.7%-pruned network against
#      a 98.5%-pruned one.
#   4. the four other criteria at exactly those keep_fractions.
# ---------------------------------------------------------------------------

set -euo pipefail

# The model is REQUIRED, deliberately. This used to default to spear_repl, and
# three bare `sbatch slurm_compare.sh` calls then all ran the same platform,
# writing to the same outputs/<model>/sparsity_curve_compare/ and the same
# checkpoint paths -- three jobs' GPU spent producing one corrupted result.
# A default that silently picks an experiment is not a convenience.
if [ $# -lt 1 ]; then
    echo "usage: sbatch slurm_compare.sh <model> [finetune_epochs] [targets]"
    echo "  e.g. sbatch slurm_compare.sh spear_repl"
    echo "       sbatch slurm_compare.sh spear_repl 210 \"50.80\""
    echo "models: spear_repl | spear_repl_resnet18 | dpap_repl | lenet | vgg9 | resnet18"
    exit 2
fi

MODEL="$1"
FINETUNE_EPOCHS="${2:-30}"
TARGETS="${3:-33.46 50.80 70 90}"

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

mkdir -p checkpoints "outputs/${MODEL}" plots logs

echo "===================================================="
echo "Comparison run: ${MODEL}"
echo "Targets: ${TARGETS}   finetune epochs: ${FINETUNE_EPOCHS}"
echo "Started: $(date)   Node: $(hostname)"
echo "===================================================="

if [ ! -f "outputs/${MODEL}/trained_model.pt" ]; then
    echo "ERROR: no baseline at outputs/${MODEL}/trained_model.pt"
    echo "Run the pretrain job for this platform first."
    exit 1
fi

# Refuse to start if another compare job is already writing to this platform's
# output directory. Two concurrent runs share sparsity_curve_compare/ and the
# per-target checkpoint paths, so the second silently overwrites the first's
# partial results and both finish looking successful.
LOCK="outputs/${MODEL}/.compare.lock"
if [ -f "$LOCK" ]; then
    OTHER=$(cat "$LOCK")
    if squeue -j "$OTHER" -h -o %i 2>/dev/null | grep -q .; then
        echo "ERROR: job ${OTHER} is already running a comparison for ${MODEL}."
        echo "Wait for it, or scancel it, then resubmit."
        exit 1
    fi
    echo "NOTE: stale lock from job ${OTHER} (no longer queued); continuing."
fi
echo "${SLURM_JOB_ID:-manual}" > "$LOCK"
trap 'rm -f "'"$LOCK"'"' EXIT

python tests/test_spear.py

# --- 1. gate pressure, hard gate -------------------------------------------
echo ""
echo "--- Stage 1/4: gate-pressure diagnostic ---"
# --fail-below makes this exit non-zero when the KL is effectively unopposed,
# so `set -e` aborts the job here rather than four hours later.
python run_sparsity_curve.py --model "${MODEL}" --diagnose-only --fail-below 1e-3

# --- 2. Bayesian curve ------------------------------------------------------
echo ""
echo "--- Stage 2/4: Bayesian accuracy-vs-sparsity curve ---"
python run_sparsity_curve.py \
    --model "${MODEL}" \
    --mode uniform \
    --tag compare \
    --targets ${TARGETS} \
    --finetune-epochs "${FINETUNE_EPOCHS}"

# --- 3. matched keep_fractions, derived not transcribed ---------------------
echo ""
echo "--- Stage 3/4: deriving matched keep_fractions ---"
KEEP_FRACTIONS=$(python run_sparsity_curve.py \
    --model "${MODEL}" --targets ${TARGETS} --emit-keep-fractions)
echo "targets        : ${TARGETS}"
echo "keep_fractions : ${KEEP_FRACTIONS}"

# --- 4. the other four criteria at exactly those widths ---------------------
echo ""
echo "--- Stage 4/4: naive / SCA / DPAP / Network Slimming ---"
python run_bio_pruning.py \
    --models "${MODEL}" \
    --criteria naive_firing_rate sca dpap network_slimming \
    --keep-fractions ${KEEP_FRACTIONS} \
    --finetune-epochs "${FINETUNE_EPOCHS}" \
    --output "outputs/bio_results_${MODEL}.csv"

echo "===================================================="
echo "Finished: $(date)"
echo "===================================================="
echo ""
echo "Results:"
echo "  Bayesian : outputs/${MODEL}/sparsity_curve_compare/summary.csv"
echo "  Others   : outputs/bio_results_${MODEL}.csv"
echo ""
echo "PUSH THESE OFF SCRATCH BEFORE THEY ARE LOST. Every headline number in"
echo "HANDOFF.md currently exists only as prose because this step was skipped."
echo ""
echo "Then check, before believing any of it:"
echo "  - gate health in the curve log (log_alpha std, saturation)"
echo "  - gamma_std in the slim_train log; near zero means an arbitrary keep-set"
