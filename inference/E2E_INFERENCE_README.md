# End-to-End Style Transfer Inference

This document explains how to use the end-to-end style transfer inference script that performs mixing style transfer starting from raw audio clips (no pre-separated stems required).

## Overview

The `inference_e2e_style_transfer.py` script provides two optimization modes:

1. **Embeddings Mode**: Optimizes TCN to match a 512-dimensional learned embedding space
2. **Features Mode**: Optimizes TCN directly on 56-dimensional hand-crafted mixing features

## Important: Audio Segment Length

**⚠️ The mixing encoder was trained on 10-second audio segments.**

- By default, the script extracts a **10-second segment** from the beginning of each audio file
- You can specify a different offset using `--segment_offset` to extract a different 10-second window
- **Do not change `--segment_duration` from 10.0s** unless you have VRAM constraints or want to experiment
- Using different segment lengths may cause:
  - VRAM overflow (longer segments)
  - Degraded quality (encoder not trained on that length)

## Features

- **Automatic Source Separation**: Uses SCNet to separate audio into 4 stems (vocals, bass, drums, other)
- **Dual Optimization Modes**: Choose between embeddings or features optimization
- **Audio Segmentation**: Automatically extracts 10-second segments (encoder constraint)
- **Comprehensive Metrics**: Tracks both embedding distance and feature distance throughout optimization
- **Complete Outputs**: Saves audio, stems, and detailed metrics

## Usage

### Quick Start with Helper Script

```bash
# Run with embeddings optimization
./run_e2e_inference.sh embeddings input.wav target.wav results/test1

# Run with features optimization
./run_e2e_inference.sh features input.wav target.wav results/test2

# With custom hyperparameters
./run_e2e_inference.sh embeddings input.wav target.wav results/test3 300 0.002
```

### Direct Python Usage

```bash
# Embeddings mode
CUDA_VISIBLE_DEVICES=1 python inference_e2e_style_transfer.py \
    --input_audio /path/to/input.wav \
    --target_audio /path/to/target.wav \
    --optimize_target embeddings \
    --output_dir results/embeddings_test \
    --num_steps 500 \
    --lr 0.001 \
    --receptive_field 2.0

# Features mode
CUDA_VISIBLE_DEVICES=1 python inference_e2e_style_transfer.py \
    --input_audio /path/to/input.wav \
    --target_audio /path/to/target.wav \
    --optimize_target features \
    --output_dir results/features_test \
    --num_steps 500 \
    --lr 0.001 \
    --receptive_field 2.0
```

## Arguments

### Required Arguments

- `--input_audio`: Path to input audio file (any format supported by soundfile)
- `--target_audio`: Path to target audio file
- `--optimize_target`: Choose `embeddings` or `features`

### Optional Arguments

- `--output_dir`: Output directory (default: `e2e_style_transfer_output`)
- `--mixing_checkpoint`: Path to mixing model checkpoint (default: `/nas/mixing-representation/checkpoints_baseline/best_model.pt`)
- `--scnet_model`: Path to SCNet checkpoint (default: `Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt`)
- `--scnet_config`: Path to SCNet config (default: `Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml`)
- `--num_steps`: Number of optimization steps (default: 500)
- `--lr`: Learning rate (default: 0.001)
- `--receptive_field`: TCN receptive field in seconds (default: 2.0)
- `--segment_duration`: Audio segment duration in seconds (default: 10.0) **⚠️ Do not change**
- `--segment_offset`: Start offset in seconds for segment extraction (default: 0.0)
- `--device`: Device to use, `cuda` or `cpu` (default: `cuda`)

**Note on segment parameters:**
- `--segment_duration` should remain at 10.0s (encoder trained on this length)
- `--segment_offset` allows you to select which 10-second window to process (e.g., `--segment_offset 30.0` processes seconds 30-40)

## Output Structure

```
{output_dir}/
├── input_original.wav           # Original input audio
├── target_original.wav          # Original target audio
├── output_transferred.wav       # Style-transferred result
├── input_stems/                 # Separated input stems
│   ├── vocals.wav
│   ├── bass.wav
│   ├── drums.wav
│   └── other.wav
├── target_stems/                # Separated target stems
│   ├── vocals.wav
│   ├── bass.wav
│   ├── drums.wav
│   └── other.wav
└── metrics.json                 # Optimization metrics
```

## Metrics JSON Format

```json
{
  "input_audio": "path/to/input.wav",
  "target_audio": "path/to/target.wav",
  "optimize_target": "embeddings",
  "hyperparameters": {
    "num_steps": 500,
    "lr": 0.001,
    "receptive_field": 2.0
  },
  "initial_embedding_distance": 0.8234,
  "final_embedding_distance": 0.3456,
  "initial_feature_distance": 0.7123,
  "final_feature_distance": 0.2890,
  "converged": true,
  "iterations": [
    {"step": 0, "embedding_distance": 0.8234, "feature_distance": 0.7123},
    {"step": 1, "embedding_distance": 0.8100, "feature_distance": 0.7050},
    ...
  ]
}
```

**Note**: Both embedding distance and feature distance are tracked regardless of optimization mode, enabling direct comparison.

## How It Works

### Pipeline

1. **Load Audio**: Load input and target audio files (auto-convert to stereo, 44.1kHz)
2. **Source Separation**: Separate both audio clips into 4 stems using SCNet
3. **Compute Target**: Extract target embedding (embeddings mode) or features (features mode)
4. **Initialize TCN**: Create TCN mixer with identity initialization
5. **Optimize**: Run gradient descent to minimize distance to target
6. **Save Results**: Save transferred audio, stems, and comprehensive metrics

### Optimization Modes Comparison

| Aspect | Embeddings Mode | Features Mode |
|--------|----------------|---------------|
| **Target Space** | 512-dim learned embedding | 56-dim hand-crafted features |
| **Model Required** | MixingStyleEncoder (loaded) | Feature extractor only |
| **Gradient Path** | Through deep CNN + FiLM | Direct through feature extraction |
| **Speed** | Moderate (includes model forward pass) | Faster (skips model) |
| **Expected Quality** | High (model trained for this) | Unknown (features not trained) |

### Mixing Features (56 dimensions)

The features include:
- **Dynamics** (24 features): RMS energy, crest factor, loudness per stem
- **Spectral** (20 features): Low/mid/high band energies, tilt, flatness per stem
- **Stereo** (12 features): ILD, correlation, mid-side ratio per stem
- **Relative Loudness** (4 features): Stem loudness vs mixture
- **Inter-Stem Masking** (4 features): Frequency domain masking between stems

All features are fully differentiable (pure PyTorch operations).

## Troubleshooting

### CUDA Out of Memory
- Use `CUDA_VISIBLE_DEVICES=1` to select a different GPU
- Reduce audio length or reduce receptive field
- Process on CPU (slower): `--device cpu`

### Source Separation Errors
- Ensure audio files are readable and have valid format
- Check that SCNet checkpoint and config paths are correct
- Verify audio is at least 1 second long

### NaN in Gradients
- Try reducing learning rate (e.g., `--lr 0.0005`)
- Check for silent or constant audio regions
- Enable gradient clipping (may need code modification)

### Poor Convergence
- Increase number of steps: `--num_steps 1000`
- Adjust learning rate (try 0.0005 to 0.002)
- Try different receptive field (1.5s to 3.0s)
- Check if audio clips are too different (e.g., different genres)

## Examples

### Basic Usage
```bash
# Simple test with default settings
./run_e2e_inference.sh embeddings \
    /nas/MUSDB18_Balanced/train/Actions\ -\ One\ Minute\ Smile/mixture.wav \
    /nas/MUSDB18_Balanced/train/Helado\ Negro\ -\ Mitad\ Del\ Mundo/mixture.wav \
    results/pair1_embeddings
```

### Comparing Both Modes
```bash
# Run same pair with both modes
INPUT="/path/to/input.wav"
TARGET="/path/to/target.wav"

# Embeddings mode
./run_e2e_inference.sh embeddings "$INPUT" "$TARGET" results/comparison_embeddings

# Features mode
./run_e2e_inference.sh features "$INPUT" "$TARGET" results/comparison_features

# Compare metrics
cat results/comparison_embeddings/metrics.json
cat results/comparison_features/metrics.json
```

### Hyperparameter Search
```bash
# Test different learning rates
for LR in 0.0005 0.001 0.002; do
    ./run_e2e_inference.sh embeddings input.wav target.wav "results/lr_${LR}" 500 $LR
done
```

## Performance Notes

- **Source separation**: ~5-10 seconds per 10-second audio clip on GPU
- **Optimization**: ~1-2 minutes for 500 steps with embeddings mode
- **Total time**: ~2-3 minutes per pair for complete pipeline

## Citation

If you use this code, please cite the original mixing representation work and the SCNet source separation model.
