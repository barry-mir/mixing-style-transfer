#!/bin/bash
# End-to-End Style Transfer Inference with Pretrained TCN
# Supports both mixing_style and fx_encoder
#
# Usage:
#   bash scripts/run_e2e_inference.sh <encoder_type> <tcn_checkpoint> <input.wav> <target.wav> <output_dir>
#
# Arguments:
#   encoder_type: 'mixing_style' or 'fx_encoder'
#   tcn_checkpoint: Path to trained TCN checkpoint
#   input.wav: Path to input audio file
#   target.wav: Path to target audio file
#   output_dir: Output directory for results
#
# Examples:
#   # MixingStyle encoder (stem-based, 512-dim)
#   bash scripts/run_e2e_inference.sh mixing_style /path/to/tcn.pt input.wav target.wav results/test1
#
#   # Fx-Encoder (mixture-based, 128-dim)
#   bash scripts/run_e2e_inference.sh fx_encoder /path/to/tcn.pt input.wav target.wav results/test2

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export PYTHONUNBUFFERED=1

# Parse arguments
ENCODER_TYPE="fx_encoder"
TCN_CHECKPOINT="/nas/mixing-representation/style_transfer/fx_encoder_plus_plus/best_model.pt"
INPUT_AUDIO="assets/song_A.wav"
TARGET_AUDIO="assets/song_B.wav"
OUTPUT_DIR="outputs/test_adv_no_cycle"

# Validate encoder type
if [[ "$ENCODER_TYPE" != "mixing_style" && "$ENCODER_TYPE" != "fx_encoder" ]]; then
    echo "Error: Invalid encoder type '$ENCODER_TYPE'"
    echo "Must be 'mixing_style' or 'fx_encoder'"
    exit 1
fi

# Check required arguments
if [[ -z "$ENCODER_TYPE" || -z "$TCN_CHECKPOINT" || -z "$INPUT_AUDIO" || -z "$TARGET_AUDIO" ]]; then
    echo "Usage: bash scripts/run_e2e_inference.sh <encoder_type> <tcn_checkpoint> <input.wav> <target.wav> [output_dir]"
    echo ""
    echo "Encoder types:"
    echo "  mixing_style - Stem-based encoder with hand-crafted features (512-dim)"
    echo "  fx_encoder   - Mixture-based Fx-Encoder++ (128-dim)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_e2e_inference.sh mixing_style /path/to/tcn.pt input.wav target.wav results/test1"
    echo "  bash scripts/run_e2e_inference.sh fx_encoder /path/to/tcn.pt input.wav target.wav results/test2"
    exit 1
fi

echo "================================"
echo "E2E Style Transfer Inference"
echo "================================"
echo "Encoder: $ENCODER_TYPE"
echo "TCN checkpoint: $TCN_CHECKPOINT"
echo "Input: $INPUT_AUDIO"
echo "Target: $TARGET_AUDIO"
echo "Output: $OUTPUT_DIR"
echo "================================"
echo ""

# Set encoder-specific arguments
if [[ "$ENCODER_TYPE" == "mixing_style" ]]; then
    ENCODER_ARGS="--encoder_checkpoint /nas/mixing-representation/checkpoints_adversarial/best_model.pt"
elif [[ "$ENCODER_TYPE" == "fx_encoder" ]]; then
    ENCODER_ARGS="--fx_encoder_model default"
fi

# Run inference
python inference/inference_e2e_style_transfer.py \
    --encoder_type "$ENCODER_TYPE" \
    --tcn_checkpoint "$TCN_CHECKPOINT" \
    --input_audio "$INPUT_AUDIO" \
    --target_audio "$TARGET_AUDIO" \
    --output_dir "$OUTPUT_DIR" \
    $ENCODER_ARGS \
    --scnet_model Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \
    --scnet_config Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml \
    --segment_duration 10.0 \
    --segment_offset 0.0 \
    --device cuda

echo ""
echo "================================"
echo "Inference complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/transferred_audio.wav (final result)"
echo "  - $OUTPUT_DIR/input_original.wav (input segment)"
echo "  - $OUTPUT_DIR/target_original.wav (target segment)"
echo "  - $OUTPUT_DIR/metadata.json (run metadata)"
echo ""
echo "Listen to the result:"
echo "  play $OUTPUT_DIR/transferred_audio.wav"
