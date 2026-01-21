# Mixing Style Representation Learning

A stem-aware contrastive framework for learning mixing style representations from audio.

## Overview

This project implements Stage 1 (Contrastive Pretraining) of the mixing style representation learning framework described in the research proposal. The system learns interpretable embeddings that capture mixing style characteristics through:

1. **Source Separation**: SCNet separates audio into 4 stems (vocals, bass, drums, other)
2. **Stem-Aware Encoding**: Band-split CNN processes 8-channel input (4 stems × 2 stereo)
3. **FiLM Conditioning**: Mixing features modulate the encoder via Feature-wise Linear Modulation
4. **Contrastive Learning**: InfoNCE loss learns representations where:
   - **Positive pairs**: Different clips from same song (same mixing style, different content)
   - **Negative pairs**: Original clip + augmented version (different mixing style, same content)

## Project Structure

```
mixing-representation/
├── src/
│   ├── model.py       # Band-split CNN encoder with FiLM conditioning
│   ├── train.py       # Training loop
│   ├── params.py      # Hyperparameters and argument parser
│   ├── data.py        # FMA dataset with SCNet inference
│   ├── loss.py        # InfoNCE contrastive loss
│   └── utils.py       # Mixing feature extraction + augmentations
├── scripts/
│   └── train.sh       # Training bash script
├── Music-Source-Separation-Training/  # SCNet model repo
├── SubSpectralNet/    # Reference for band-split architecture
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure SCNet model checkpoint exists:
```bash
ls Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt
```

3. Verify FMA dataset location:
```bash
ls /nas/FMA/fma_full/
```

## Training

### Quick Start

Run training with default parameters:
```bash
bash scripts/train.sh
```

### Custom Training

```bash
python src/train.py \
    --data_path /nas/FMA/fma_full/ \
    --batch_size 24 \
    --num_epochs 100 \
    --learning_rate 2e-4 \
    --temperature 0.1 \
    --device cuda
```

### Key Hyperparameters

- `--batch_size`: Batch size (default: 24)
- `--learning_rate`: Learning rate for AdamW (default: 2e-4)
- `--temperature`: Temperature for InfoNCE loss (default: 0.1)
- `--clip_duration`: Duration of audio clips in seconds (default: 10.0)
- `--n_mels`: Number of mel bands (default: 128)
- `--encoder_dim`: Encoder embedding dimension (default: 768)
- `--aug_prob`: Probability of applying augmentations (default: 0.5)
- `--aug_gain_range`: Gain augmentation range in dB (default: ±9.0)

## Model Architecture

### Band-Split CNN Encoder
- **Input**: 8-channel mel-spectrogram (4 stems × 2 stereo channels)
- **Processing**: Split into overlapping sub-bands (size=20, overlap=10)
- **Each sub-band**: Conv2D → BatchNorm → FiLM → ReLU → MaxPool → Conv2D → 2D output (no flatten)
- **Aggregation**:
  - Concatenate all sub-band 2D features
  - **Temporal attention pooling** learns to weight important time frames
- **Output**: 768-dimensional embedding

### FiLM Conditioning
- **Input**: Mixing features (dynamics, spectral, stereo, masking)
- **Processing**: MLP → FiLM parameters (γ, β)
- **Application**: Modulates sub-band CNN activations

### Mixing Features
1. **Dynamics**: RMS, crest factor, LUFS loudness, relative loudness
2. **Spectral**: Band energies, spectral tilt, flatness
3. **Stereo**: ILD, correlation, mid-side ratio
4. **Masking**: Inter-stem dominance metrics

## Data Pipeline

1. Load audio from FMA dataset
2. Crop two different 10-second clips (positive pair)
3. Separate both clips into 4 stems using SCNet (no-grad, in-memory)
4. Augment first clip stems to create negative pair
5. Extract mixing features for all three versions
6. Compute 8-channel mel-spectrograms
7. Return triplet: (anchor, positive, negative)

## Augmentations

Applied to create negative pairs with different mixing styles:
- Gain imbalance (±9 dB)
- Spectral tilt (EQ filtering)
- Dynamic range compression
- Bandwidth limiting
- Stereo reverb

## Output

### Checkpoints
Saved to `checkpoints/`:
- `checkpoint_epoch_X.pt`: Regular checkpoints every 5 epochs
- `best_model.pt`: Best model based on training loss
- `final_model.pt`: Final model after training

### Logs
TensorBoard logs saved to `logs/`:
```bash
tensorboard --logdir logs/
```

## Implementation Notes

### Key Differences from Proposal

1. **Source Separation**: Uses actual SCNet model from Music-Source-Separation-Training repo
2. **Encoder**: Band-split CNN adapted from SubSpectralNet (PyTorch implementation)
3. **Training**: Stage 1 only (contrastive pretraining), Stage 2 (regression) not implemented yet
4. **Contrastive Strategy**:
   - Positive = same mixing style (different clips from same song)
   - Negative = different mixing style (augmented version)

### Performance Considerations

- SCNet inference is performed on-the-fly (no pre-separation)
- Uses `torch.no_grad()` for SCNet to save memory
- Multi-worker data loading for efficiency
- Mixed precision training can be added for speedup

## Citation

```
@article{cheng2024mixing,
  title={Mixing Style Representation Learning: A Stem-Aware Contrastive Framework with Structured Mix Features},
  author={Cheng, Barry},
  year={2024}
}
```

## License

See LICENSE file for details.
