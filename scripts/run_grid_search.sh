#!/bin/bash
#
# Run TCN hyperparameter grid search
#

cd "$(dirname "$0")"

echo "=========================================="
echo "TCN Hyperparameter Grid Search"
echo "=========================================="
echo ""
echo "Grid configuration:"
echo "  Optimizers: Adam, AdamW"
echo "  Learning rates: 0.0005, 0.001, 0.002"
echo "  Steps: 300, 500"
echo "  Hidden channels: 64, 128"
echo "  Receptive fields: 1.5s, 2.0s, 3.0s"
echo ""
echo "Total: 2×3×2×2×3 = 72 configurations"
echo "Test pairs: 5"
echo "Total experiments: 360"
echo ""
echo "Estimated time: ~30-50 hours"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=1 python ../inference/grid_search_tcn.py \
    --checkpoint /nas/mixing-representation/checkpoints_baseline/best_model.pt \
    --musdb_path /nas/MUSDB18 \
    --output_file tcn_grid_search_results.json \
    --num_pairs 5 \
    --seed 42

echo ""
echo "=========================================="
echo "Grid Search Complete!"
echo "=========================================="
