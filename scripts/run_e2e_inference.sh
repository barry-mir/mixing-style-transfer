#!/bin/bash
# Helper script to run end-to-end style transfer inference

# Usage examples:
# ./run_e2e_inference.sh embeddings input.wav target.wav output_dir
# ./run_e2e_inference.sh features input.wav target.wav output_dir

# set -e

# if [ "$#" -lt 4 ]; then
#     echo "Usage: $0 <optimize_target> <input_audio> <target_audio> <output_dir> [num_steps] [lr]"
#     echo ""
#     echo "Arguments:"
#     echo "  optimize_target: 'embeddings' or 'features'"
#     echo "  input_audio: Path to input audio file"
#     echo "  target_audio: Path to target audio file"
#     echo "  output_dir: Output directory for results"
#     echo "  num_steps: (optional) Number of optimization steps (default: 500)"
#     echo "  lr: (optional) Learning rate (default: 0.001)"
#     echo ""
#     echo "Example:"
#     echo "  $0 embeddings /path/to/input.wav /path/to/target.wav results/test1"
#     echo "  $0 features /path/to/input.wav /path/to/target.wav results/test2 300 0.002"
#     exit 1
# fi

OPTIMIZE_TARGET="embeddings"
INPUT_AUDIO="../assets/song_A.wav"
TARGET_AUDIO="../assets/song_B.wav"
OUTPUT_DIR="../outputs/test_emb_0121"
NUM_STEPS=500
LR=0.001
SEGMENT_DURATION=10.0
SEGMENT_OFFSET=0.0

echo "=========================================="
echo "End-to-End Style Transfer"
echo "=========================================="
echo "Optimization target: $OPTIMIZE_TARGET"
echo "Input audio: $INPUT_AUDIO"
echo "Target audio: $TARGET_AUDIO"
echo "Output directory: $OUTPUT_DIR"
echo "Num steps: $NUM_STEPS"
echo "Learning rate: $LR"
echo "Segment: ${SEGMENT_DURATION}s (offset: ${SEGMENT_OFFSET}s)"
echo "=========================================="
echo ""

# Run inference on GPU 1
# Note: Audio is segmented to 10s by default (encoder trained on 10s clips)
CUDA_VISIBLE_DEVICES=1 python ../inference/inference_e2e_style_transfer.py \
    --input_audio "$INPUT_AUDIO" \
    --target_audio "$TARGET_AUDIO" \
    --optimize_target "$OPTIMIZE_TARGET" \
    --output_dir "$OUTPUT_DIR" \
    --num_steps "$NUM_STEPS" \
    --lr "$LR" \
    --receptive_field 2.0 \
    --segment_duration "$SEGMENT_DURATION" \
    --segment_offset "$SEGMENT_OFFSET" \
    --device cuda

echo ""
echo "Done! Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - ${OUTPUT_DIR}/output_transferred.wav  (style-transferred audio)"
echo "  - ${OUTPUT_DIR}/metrics.json            (optimization metrics)"
echo "  - ${OUTPUT_DIR}/input_original.wav      (original input)"
echo "  - ${OUTPUT_DIR}/target_original.wav     (original target)"
echo "  - ${OUTPUT_DIR}/input_stems/            (separated input stems)"
echo "  - ${OUTPUT_DIR}/target_stems/           (separated target stems)"
