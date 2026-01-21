#!/bin/bash
#
# Quick TCN style transfer test - 1 pair, 200 steps
#

cd "$(dirname "$0")"

echo "========================================"
echo "Quick TCN Style Transfer Test"
echo "========================================"
echo ""
echo "Approach: Differentiable optimization"
echo "  - TCN processes 8 channels (4 stems × stereo)"
echo "  - Gradient descent on embedding distance"
echo "  - 200 optimization steps"
echo "  - ~2 second receptive field"
echo ""
echo "Expected runtime: ~5-10 minutes"
echo "========================================"
echo ""

CUDA_VISIBLE_DEVICES=1 python ../inference/test_tcn_style_transfer.py \
    --checkpoint /nas/mixing-representation/checkpoints_baseline/best_model.pt \
    --musdb_path /nas/MUSDB18_Balanced \
    --output_dir tcn_style_transfer_results_quick \
    --num_pairs 10 \
    --num_steps 200 \
    --lr 0.0001 \
    --segment_duration 10.0 \
    --receptive_field 2.0 \
    --seed 42

echo ""
echo "========================================"
echo "Quick Test Complete!"
echo "========================================"
