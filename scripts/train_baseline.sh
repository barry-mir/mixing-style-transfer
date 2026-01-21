#!/bin/bash

# Baseline Training Script - Simple InfoNCE without augmentation
# Just 2 random clips per song as positive pairs

cd "$(dirname "$0")/.."

# Set Python path to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Optimize for data loading performance
NUM_CPUS=$(nproc)
# NUM_WORKERS=$((NUM_CPUS - 2))  # Leave 2 cores for main process
NUM_WORKERS=12

# Set threading environment variables
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "=========================================="
echo "BASELINE TRAINING MODE"
echo "=========================================="
echo "System has $NUM_CPUS CPU cores"
echo "Using $NUM_WORKERS DataLoader workers"
echo "Main process threads: 2 (OMP/MKL/OPENBLAS)"
echo ""
echo "Baseline setup:"
echo "  - num_mix_variants=1 (no augmentation)"
echo "  - num_segments=2 (2 clips per song)"
echo "  - Positive: same song, different clips"
echo "  - Negative: different songs"
echo "=========================================="

python src/train.py \
    --use_preseparated \
    --separated_path /ssd2/barry/fma_large_separated/ \
    --data_path /nas/FMA/fma_large/ \
    --scnet_model_path Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \
    --scnet_config_path Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml \
    --batch_size 100 \
    --num_epochs 100 \
    --learning_rate 2e-4 \
    --temperature 0.1 \
    --clip_duration 10.0 \
    --n_fft 2048 \
    --hop_length 512 \
    --n_mels 80 \
    --encoder_dim 512 \
    --feature_dim 128 \
    --band_split_size 16 \
    --band_overlap 8 \
    --num_songs_per_batch 80 \
    --num_mix_variants 1 \
    --num_segments 2 \
    --aug_prob 0.0 \
    --aug_gain_range 0.0 \
    --num_workers $NUM_WORKERS \
    --log_interval 10 \
    --save_interval 5 \
    --checkpoint_dir /nas/mixing-representation/checkpoints_baseline_large/ \
    --log_dir /nas/mixing-representation/logs_baseline_large/ \
    --device cuda \
    --seed 42
