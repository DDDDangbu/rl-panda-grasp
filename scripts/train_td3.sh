#!/bin/bash
# Train TD3 + HER with curriculum learning
# Usage: bash scripts/train_td3.sh

set -e

echo "=========================================="
echo "Training: TD3 + HER + Curriculum Learning"
echo "=========================================="

python -m training.train \
    --config configs/td3_her.yaml \
    --run_name td3_her_curriculum

echo ""
echo "Training complete!"
echo "  Model: results/models/td3_her_curriculum/"
echo "  Logs:  results/logs/td3_her_curriculum/"
echo "  View:  tensorboard --logdir results/logs"
