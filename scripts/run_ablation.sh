#!/bin/bash
# Run all ablation study experiments
# Usage: bash scripts/run_ablation.sh
# Usage (shorter): bash scripts/run_ablation.sh --timesteps 200000

set -e

echo "=========================================="
echo "Running Ablation Study"
echo "=========================================="

python -m training.ablation "$@"

echo ""
echo "Generating comparison plots..."
python -m evaluation.plot_results --log_dir results/logs --output results/plots

echo ""
echo "All done!"
echo "  Plots: results/plots/"
echo "  View:  tensorboard --logdir results/logs"
