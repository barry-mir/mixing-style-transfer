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

    # Adversarial training parameters
    parser.add_argument('--use_adversarial', action='store_true', default=False,
                        help='Enable adversarial training to remove song identity')
    parser.add_argument('--adversarial_lambda', type=float, default=1.0,
                        help='Final weight for adversarial loss (default: 1.0)')
    parser.add_argument('--initial_adversarial_lambda', type=float, default=0.0,
                        help='Initial weight for adversarial loss, ramps up to adversarial_lambda (default: 0.0)')
    parser.add_argument('--adversarial_warmup_steps', type=int, default=2000,
                        help='Number of steps before starting adversarial training (default: 2000)')
    parser.add_argument('--fixed_grl_lambda', type=float, default=None,
                        help='Fix GRL lambda to a constant value instead of using schedule (e.g., 1.0). If None, uses DANN schedule.')
    parser.add_argument('--song_id_cache_path', type=str,
                        default='/ssd2/barry/fma_song_identity_embeddings.pt',
                        help='Path to pre-computed song identity embeddings cache')
    parser.add_argument('--discriminator_hidden_dim', type=int, default=512,
                        help='Hidden dimension for song identity discriminator (default: 512)')
    parser.add_argument('--discriminator_dropout', type=float, default=0.3,
                        help='Dropout probability for discriminator (default: 0.3)')
    parser.add_argument('--discriminator_lr', type=float, default=None,
                        help='Learning rate for discriminator. If None, uses same as encoder. Set lower (e.g., 1e-5) to weaken discriminator.')
    parser.add_argument('--discriminator_noise', type=float, default=0.0,
                        help='Add Gaussian noise to embeddings before discriminator (e.g., 0.01) to weaken discriminator. Default: 0.0 (no noise)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--weights_only', action='store_true', default=False,
                        help='Load only model weights from checkpoint, reset training state (start from epoch 0)')

    args = parser.parse_args()
    return args
