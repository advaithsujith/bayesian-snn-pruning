#!/bin/bash
#SBATCH --job-name=spear_r18_pretrain
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spear_r18_%j.out
#SBATCH --error=logs/spear_r18_%j.err

# ---------------------------------------------------------------------------
# SPEAR replication, ResNet18 arm: pretrain the baseline.
#
# Second architecture of the head-to-head against Xie et al. (arXiv
# 2507.02945). Identical training recipe to slurm_spear.sh -- SPEAR states one
# recipe for all static datasets and both architectures -- so only the
# architecture differs. Provenance in docs/replication_targets.md section 4.
#
# Target row: 39.2% SynOps, 30.3% params, 92.78% top-1.
#
# Timing: the VGG16 arm ran 210 epochs at 25.3 s/epoch, ~1.5h. ResNet18 at
# T=4 has more layers and residual adds but similar spatial sizes; expect
# somewhere around 2-4h. Never measured, hence the generous 24h.
#
# BEFORE SUBMITTING:
#   git pull
#   source venv/bin/activate
#   python tests/test_spear.py
#   python tests/test_vggstyle.py
#   python tests/test_ranked_pruning.py
#   python tests/test_synops.py
#
# --pretrain-only for the same reason as the VGG16 arm: beta_max=0.01 is a
# placeholder carried over from dpap_repl and does not transfer. ResNet18 is
# the worst case for that, since half its gate units sit on non-prunable
# layers and are excluded from the KL entirely (HANDOFF.md Session 3, bug #2).
# Read the diagnostic on THIS baseline before spending gate hours:
#   python run_sparsity_curve.py --model spear_repl_resnet18 --diagnose-only
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

mkdir -p checkpoints outputs/spear_repl_resnet18 plots logs

echo "===================================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Fail fast on the login node's schedule rather than 200 epochs into the queue.
python tests/test_spear.py

python run_all.py --model spear_repl_resnet18 --pretrain-only

echo "===================================================="
echo "Job finished: $(date)"
echo "===================================================="
echo ""
echo "Baseline saved to outputs/spear_repl_resnet18/trained_model.pt"
echo ""
echo "NOTE: SPEAR publishes no unpruned baseline for ResNet18 either, and"
echo "unlike VGG16 there is no SCA row to borrow as a reference. Their pruned"
echo "92.78% implies a dense baseline around 93%. This project's own resnet18"
echo "experiment reached 89.26%, but at T=25 rather than T=4."
echo ""
echo "Next:"
echo "  1. push outputs/spear_repl_resnet18 off scratch"
echo "  2. python measure_baseline_synops.py --model spear_repl_resnet18 --batches 8"
echo "  3. set reuse_pretrained=True in get_spear_repl_resnet18_config()"
echo "  4. python run_sparsity_curve.py --model spear_repl_resnet18 --diagnose-only"
