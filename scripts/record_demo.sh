#!/bin/bash
# Record demo videos of a trained agent
# Usage: bash scripts/record_demo.sh
# Usage: bash scripts/record_demo.sh results/models/sac_her_curriculum/best_model.zip

set -e

MODEL=${1:-"results/models/sac_her_curriculum/best_model.zip"}

echo "=========================================="
echo "Recording Demo Videos"
echo "Model: $MODEL"
echo "=========================================="

# Record at full difficulty
python -m evaluation.record_video \
    --model "$MODEL" \
    --output results/videos \
    --episodes 5 \
    --difficulty 1.0

# Record across difficulty levels
python -m evaluation.record_video \
    --model "$MODEL" \
    --output results/videos \
    --sweep

echo ""
echo "Videos saved to: results/videos/"
