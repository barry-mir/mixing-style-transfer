#!/bin/bash

# Training script for Mixing Style Representation Learning (Stage 1)
# Optimized for maximum CPU utilization

cd "$(dirname "$0")/.."

# Set Python path to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Optimize for data loading performance
# - Use all CPU cores for DataLoader workers
# - Set optimal thread counts for torch/numpy
NUM_CPUS=$(nproc)
NUM_WORKERS=$((NUM_CPUS - 2))  # Leave 2 cores for main process

# Set threading environment variables
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "System has $NUM_CPUS CPU cores"
echo "Using $NUM_WORKERS DataLoader workers"
echo "Main process threads: 2 (OMP/MKL/OPENBLAS)"

python src/train.py \
    --use_preseparated \
    --separated_path /ssd2/barry/fma_separated/ \
    --data_path /ssd2/barry/fma_full/ \
    --scnet_model_path Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \
    --scnet_config_path Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 2e-4 \
    --temperature 0.1 \
    --clip_duration 10.0 \
    --n_fft 2048 \
    --hop_length 512 \
    --n_mels 128 \
    --encoder_dim 512 \
    --feature_dim 128 \
    --band_split_size 16 \
    --band_overlap 8 \
    --aug_prob 0.5 \
    --aug_gain_range 9.0 \
    --num_workers $NUM_WORKERS \
    --use_amp \
    --log_interval 10 \
    --save_interval 5 \
    --checkpoint_dir /nas/mixing-representation/checkpoints/ \
    --log_dir /nas/mixing-representation/logs/ \
    --device cuda \
    --seed 42
