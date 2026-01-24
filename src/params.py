"""
Hyperparameters and argument parser for mixing style representation learning.
"""

import argparse


def get_params():
    """Parse command line arguments and return hyperparameters."""
    parser = argparse.ArgumentParser(
        description='Mixing Style Representation Learning - Stage 1: Contrastive Pretraining'
    )

    # Dataset parameters
    parser.add_argument('--separated_path', type=str, default='/nas/FMA/fma_separated/',
                        help='Path to pre-separated stems directory')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--clip_duration', type=float, default=10.0,
                        help='Duration of audio clips in seconds')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Audio preprocessing
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT size')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop length for STFT')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel bands')

    # Model architecture
    parser.add_argument('--encoder_dim', type=int, default=768,
                        help='Encoder embedding dimension')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Mixing feature embedding dimension')
    parser.add_argument('--band_split_size', type=int, default=20,
                        help='Sub-spectrogram band size')
    parser.add_argument('--band_overlap', type=int, default=10,
                        help='Overlap between sub-spectrograms')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')

    # Contrastive learning parameters
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for InfoNCE loss')
    parser.add_argument('--num_segments', type=int, default=2,
                        help='Number of temporal segments per song (for positive pairs)')

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='Directory for tensorboard logs')

    # Device and precision
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use mixed precision training (FP16) to reduce VRAM')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    return args
