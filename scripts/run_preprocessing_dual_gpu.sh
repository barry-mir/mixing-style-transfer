#!/bin/bash
#
# Run preprocessing on both GPUs in parallel
# CUDA:0 = RTX 3080 (10GB) - Conservative settings
# CUDA:1 = RTX 3090 (24GB) - Aggressive settings
#
# Usage: bash scripts/run_preprocessing_dual_gpu.sh
#

set -e

PROJECT_ROOT="/home/barrycheng/mixing-representation"
cd "$PROJECT_ROOT"

# Output directories for logs
LOG_DIR="preprocessing_logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Dual GPU Preprocessing - FMA Dataset"
echo "========================================"
echo "CUDA:0 = RTX 3090 (24GB) - Aggressive"
echo "CUDA:1 = RTX 3080 (10GB) - Conservative"
echo ""

# Kill any existing preprocessing jobs
echo "Checking for existing jobs..."
pkill -f "preprocess_fma_separation_fast.py" || true
sleep 2

# Start GPU 0 (RTX 3090 - 24GB) with aggressive GPU settings, moderate CPU
echo ""
echo "Starting GPU 0 (RTX 3090 - 24GB)..."
CUDA_VISIBLE_DEVICES=0 nohup python scripts/preprocess_fma_separation_fast.py \
    --input_dir /nas/FMA/fma_full/ \
    --output_dir /nas/FMA/fma_separated/ \
    --batch_size 8 \
    --num_workers 6 \
    --inference_batch_size 20 \
    --mp3_workers 6 \
    --bitrate 192k \
    --max_duration 300 \
    --skip_existing \
    > "$LOG_DIR/gpu0_rtx3090.log" 2>&1 &

GPU0_PID=$!
echo "  → GPU 0 started with PID: $GPU0_PID"
echo "  → Log: $LOG_DIR/gpu0_rtx3090.log"

# Start GPU 1 (RTX 3080 - 10GB) with very conservative settings
echo ""
echo "Starting GPU 1 (RTX 3080 - 10GB)..."
CUDA_VISIBLE_DEVICES=1 nohup python scripts/preprocess_fma_separation_fast.py \
    --input_dir /nas/FMA/fma_full/ \
    --output_dir /nas/FMA/fma_separated/ \
    --batch_size 2 \
    --num_workers 4 \
    --inference_batch_size 8 \
    --mp3_workers 4 \
    --bitrate 192k \
    --max_duration 300 \
    --skip_existing \
    > "$LOG_DIR/gpu1_rtx3080.log" 2>&1 &

GPU1_PID=$!
echo "  → GPU 1 started with PID: $GPU1_PID"
echo "  → Log: $LOG_DIR/gpu1_rtx3080.log"

echo ""
echo "========================================"
echo "Both GPUs are now processing!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  GPU 0: tail -f $LOG_DIR/gpu0_rtx3090.log"
echo "  GPU 1: tail -f $LOG_DIR/gpu1_rtx3080.log"
echo "  GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "Check processed tracks:"
echo "  watch -n 60 'find /nas/FMA/fma_separated/ -type d -mindepth 1 | wc -l'"
echo ""
echo "Stop processing:"
echo "  kill $GPU0_PID $GPU1_PID"
echo ""
echo "PIDs saved to: $LOG_DIR/pids.txt"
echo "$GPU0_PID" > "$LOG_DIR/pids.txt"
echo "$GPU1_PID" >> "$LOG_DIR/pids.txt"
