#!/bin/bash
# Train SAC + HER with curriculum learning
# Usage: bash scripts/train_sac.sh

set -e

echo "=========================================="
echo "Training: SAC + HER + Curriculum Learning"
echo "=========================================="

python -m training.train \
    --config configs/sac_her.yaml \
    --run_name sac_her_curriculum

echo ""
echo "Training complete!"
echo "  Model: results/models/sac_her_curriculum/"
echo "  Logs:  results/logs/sac_her_curriculum/"
echo "  View:  tensorboard --logdir results/logs"
