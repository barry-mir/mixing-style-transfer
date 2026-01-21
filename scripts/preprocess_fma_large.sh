#!/bin/bash
#
# Preprocess FMA Large dataset - already 30 seconds, just separate stems
#
# FMA Large is already cropped to 30 seconds, so we only need to:
# 1. Separate stems using SCNet
# 2. Save as MP3 format
#
# Usage:
#   bash scripts/preprocess_fma_large.sh
#

cd "$(dirname "$0")/.."

echo "========================================"
echo "FMA Large Dataset Preprocessing"
echo "========================================"
echo ""
echo "Input: /nas/FMA/fma_large"
echo "Output: /nas/FMA/fma_large_separated"
echo ""
echo "Notes:"
echo "  - FMA Large already has 30s clips"
echo "  - No cropping needed"
echo "  - Using fast batched separation"
echo "========================================"
echo ""

python scripts/preprocess_fma_separation_fast.py \
    --input_dir /nas/FMA/fma_large \
    --output_dir /nas/FMA/fma_large_separated \
    --scnet_model Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \
    --scnet_config Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml \
    --bitrate 192k \
    --device cuda \
    --batch_size 8 \
    --num_workers 8 \
    --inference_batch_size 8 \
    --mp3_workers 8 \
    --skip_existing

echo ""
echo "========================================"
echo "Preprocessing Complete!"
echo "========================================"
