"""
End-to-End Style Transfer Inference with Pretrained TCN

Performs style transfer using a pretrained TCN mixer model.
Supports two encoder types:
  1. mixing_style: Stem-based encoder with hand-crafted features (512-dim)
  2. fx_encoder: Mixture-based Fx-Encoder++ (128-dim)

No optimization - just forward pass through pretrained model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import MixingStyleEncoder
from mixing_utils import MixingFeatureExtractor
from tcn_mixer import TCNMixer, TCNFiLMGenerator
from data import SCNetSeparator


def load_audio(file_path, target_sr=44100, segment_duration=10.0, segment_offset=0.0):
    """
    Load audio file and ensure stereo format at target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default 44100)
        segment_duration: Duration of segment to extract in seconds (default 10.0)
        segment_offset: Offset in seconds where to start extraction (default 0.0)

    Returns:
        audio: (2, T) tensor where T = segment_duration * target_sr
    """
    audio, sr = sf.read(file_path)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr).T

    # Ensure stereo (2, T)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])  # Duplicate mono to stereo
    else:
        audio = audio.T  # (T, 2) → (2, T)

    # Ensure exactly 2 channels
    if audio.shape[0] != 2:
        audio = audio[:2, :]  # Take first 2 channels

    # Extract segment
    target_length = int(segment_duration * target_sr)
    start_sample = int(segment_offset * target_sr)
    end_sample = start_sample + target_length

    # Crop or pad to exact length
    if start_sample >= audio.shape[1]:
        # Offset beyond audio length, return zeros
        audio = np.zeros((2, target_length), dtype=np.float32)
    elif end_sample > audio.shape[1]:
        # Audio shorter than requested, pad with zeros
        segment = audio[:, start_sample:]
        padding = np.zeros((2, target_length - segment.shape[1]), dtype=np.float32)
        audio = np.concatenate([segment, padding], axis=1)
    else:
        # Normal case: extract segment
        audio = audio[:, start_sample:end_sample]

    return torch.from_numpy(audio).float()


def compute_mixing_embedding(
    mixture_audio,
    stems_dict,
    model,
    feature_extractor,
    device,
    encoder_type='mixing_style'
):
    """
    Compute mixing embedding using the encoder.

    Args:
        mixture_audio: (2, T) tensor
        stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other'], values (2, T)
        model: Encoder instance (MixingStyleEncoder or FxEncoderPlusPlus)
        feature_extractor: MixingFeatureExtractor instance (None for fx_encoder)
        device: torch device
        encoder_type: 'mixing_style' or 'fx_encoder'

    Returns:
        embedding: (embed_dim,) tensor (512 for mixing_style, 128 for fx_encoder)
    """
    with torch.no_grad():
        if encoder_type == 'fx_encoder':
            # Fx-Encoder uses mixture directly
            mixture_device = mixture_audio.unsqueeze(0).to(device)  # (1, 2, T)
            embedding = model.get_fx_embedding(mixture_device)  # (1, 128)
            embedding = embedding.squeeze(0)  # (128,)
        else:  # mixing_style
            stems_cpu = {k: v.detach().cpu() if v.is_cuda else v for k, v in stems_dict.items()}

            # Extract features on CPU
            mixing_features = feature_extractor.extract_all_features(stems_cpu)

            # Move to device for model
            stems_device = {k: v.unsqueeze(0).to(device) for k, v in stems_cpu.items()}
            mixing_features = mixing_features.unsqueeze(0).to(device)

            embedding = model(stems_device, mixing_features)
            embedding = embedding.squeeze(0)  # (512,)

        return embedding


def apply_style_transfer(
    tcn,
    film_generator,
    stems_input,
    target_embedding,
    input_embedding,
    device
):
    """
    Apply style transfer using pretrained TCN and FiLM generator.

    Args:
        tcn: TCNMixer instance
        film_generator: TCNFiLMGenerator instance
        stems_input: Dict with separated stems for input audio
        target_embedding: Target style embedding
        input_embedding: Input content embedding
        device: torch device

    Returns:
        dict with keys: processed_stems, processed_mixture
    """
    tcn.eval()
    film_generator.eval()

    with torch.no_grad():
        # Move input stems to device
        input_stems = {k: v.to(device) for k, v in stems_input.items()}

        # Stack input stems: (4 stems × 2 channels, T) → (1, 8, T)
        stem_order = ['vocals', 'bass', 'drums', 'other']
        input_stacked = torch.cat([input_stems[name] for name in stem_order], dim=0).unsqueeze(0)  # (1, 8, T)

        # Concatenate embeddings: input + target
        concat_embedding = torch.cat([input_embedding.unsqueeze(0), target_embedding.unsqueeze(0)], dim=1)  # (1, 2*embed_dim)

        # Generate FiLM parameters
        film_params = film_generator(concat_embedding)  # List of dicts

        # Forward pass: apply TCN with FiLM
        transferred_stacked = tcn(input_stacked, film_params=film_params)  # (1, 8, T)

        # Split back into stems
        transferred_stems = {}
        for i, stem_name in enumerate(stem_order):
            transferred_stems[stem_name] = transferred_stacked[0, i*2:(i+1)*2, :].cpu()  # (2, T)

        # Sum to create mixture
        transferred_mixture = sum(transferred_stems.values())  # (2, T)

    return {
        'processed_stems': transferred_stems,
        'processed_mixture': transferred_mixture
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='End-to-End Style Transfer with Pretrained TCN')
    parser.add_argument('--input_audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--target_audio', type=str, required=True, help='Path to target audio file')

    # Encoder selection
    parser.add_argument('--encoder_type', type=str, default='mixing_style',
                        choices=['mixing_style', 'fx_encoder'],
                        help='Encoder type: mixing_style (stem-based) or fx_encoder (mixture-based)')

    # Model paths
    parser.add_argument('--encoder_checkpoint', type=str,
                        default='/nas/mixing-representation/checkpoints_adversarial/best_model.pt',
                        help='Path to trained encoder checkpoint (mixing_style only)')
    parser.add_argument('--tcn_checkpoint', type=str, required=True,
                        help='Path to trained TCN checkpoint')
    parser.add_argument('--fx_encoder_model', type=str, default='default',
                        help='Fx-Encoder model name (default, or custom path)')
    parser.add_argument('--scnet_model', type=str,
                        default='../Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt',
                        help='Path to SCNet model checkpoint')
    parser.add_argument('--scnet_config', type=str,
                        default='../Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml',
                        help='Path to SCNet config file')

    # Feature extraction (mixing_style only)
    parser.add_argument('--use_detailed_spectral', action='store_true', default=False,
                        help='Use detailed spectral features (mixing_style only)')
    parser.add_argument('--n_spectral_bins', type=int, default=32,
                        help='Number of spectral bins for detailed spectral features')

    # Audio segment parameters
    parser.add_argument('--segment_duration', type=float, default=10.0,
                        help='Audio segment duration in seconds (default: 10.0)')
    parser.add_argument('--segment_offset', type=float, default=0.0,
                        help='Segment offset in seconds (default: 0.0)')

    # Output
    parser.add_argument('--output_dir', type=str, default='style_transfer_output',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("End-to-End Style Transfer with Pretrained TCN")
    print("="*80)
    print(f"Encoder type: {args.encoder_type}")
    print(f"Input audio: {args.input_audio}")
    print(f"Target audio: {args.target_audio}")
    print(f"TCN checkpoint: {args.tcn_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Segment duration: {args.segment_duration}s (offset: {args.segment_offset}s)")
    print(f"Device: {device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stems_dir = output_dir / 'input_stems'
    target_stems_dir = output_dir / 'target_stems'
    input_stems_dir.mkdir(exist_ok=True)
    target_stems_dir.mkdir(exist_ok=True)

    # Load audio files (extract segment)
    print(f"Loading audio files ({args.segment_duration}s segment from offset {args.segment_offset}s)...")
    input_audio = load_audio(args.input_audio, segment_duration=args.segment_duration, segment_offset=args.segment_offset)
    target_audio = load_audio(args.target_audio, segment_duration=args.segment_duration, segment_offset=args.segment_offset)
    print(f"  Input: {input_audio.shape} ({input_audio.shape[1]/44100:.2f}s)")
    print(f"  Target: {target_audio.shape} ({target_audio.shape[1]/44100:.2f}s)")
    print()

    # Save original audio
    sf.write(str(output_dir / 'input_original.wav'), input_audio.numpy().T, 44100)
    sf.write(str(output_dir / 'target_original.wav'), target_audio.numpy().T, 44100)

    # Initialize SCNet separator
    print("Initializing SCNet separator...")
    separator = SCNetSeparator(
        model_path=args.scnet_model,
        config_path=args.scnet_config,
        device=device
    )
    print()

    # Separate input audio
    print("Separating input audio...")
    input_stems = separator.separate(input_audio.to(device))
    print("  Extracted stems: vocals, bass, drums, other")

    # Save input stems
    for stem_name, stem_audio in input_stems.items():
        sf.write(str(input_stems_dir / f'{stem_name}.wav'), stem_audio.cpu().numpy().T, 44100)
    print()

    # Separate target audio
    print("Separating target audio...")
    target_stems = separator.separate(target_audio.to(device))
    print("  Extracted stems: vocals, bass, drums, other")

    # Save target stems
    for stem_name, stem_audio in target_stems.items():
        sf.write(str(target_stems_dir / f'{stem_name}.wav'), stem_audio.cpu().numpy().T, 44100)
    print()

    # Initialize feature extractor (for mixing_style) or None (for fx_encoder)
    if args.encoder_type == 'mixing_style':
        print("Initializing mixing feature extractor...")
        feature_extractor = MixingFeatureExtractor(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            use_detailed_spectral=args.use_detailed_spectral,
            n_spectral_bins=args.n_spectral_bins
        )
        print()
    else:
        feature_extractor = None

    # Load encoder
    print(f"Loading encoder ({args.encoder_type})...")
    if args.encoder_type == 'mixing_style':
        # Load mixing model
        mixing_model = MixingStyleEncoder(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            split_size=16,
            overlap=8,
            channels=8,
            embed_dim=512,
            feature_dim=64
        ).to(device)

        checkpoint = torch.load(args.encoder_checkpoint, map_location=device)
        mixing_model.load_state_dict(checkpoint['model_state_dict'])
        mixing_model.eval()
        print(f"  Model loaded from epoch {checkpoint['epoch']}")
        encoder_embed_dim = 512

    elif args.encoder_type == 'fx_encoder':
        # Import and load Fx-Encoder
        sys.path.insert(0, str(Path(__file__).parent.parent / 'Fx-Encoder_PlusPlus'))
        from fxencoder_plusplus import load_model

        mixing_model = load_model(model_name=args.fx_encoder_model, device=device)
        mixing_model.eval()
        print("  Fx-Encoder loaded successfully")
        encoder_embed_dim = 128

    print()

    # Compute embeddings
    print("Computing embeddings...")
    input_mixture = sum(input_stems.values())
    target_mixture = sum(target_stems.values())

    input_embedding = compute_mixing_embedding(
        input_mixture,
        input_stems,
        mixing_model,
        feature_extractor,
        device,
        encoder_type=args.encoder_type
    )
    target_embedding = compute_mixing_embedding(
        target_mixture,
        target_stems,
        mixing_model,
        feature_extractor,
        device,
        encoder_type=args.encoder_type
    )

    print(f"  Input embedding: {input_embedding.shape}")
    print(f"  Target embedding: {target_embedding.shape}")
    print()

    # Compute distance
    input_emb_norm = F.normalize(input_embedding, p=2, dim=0)
    target_emb_norm = F.normalize(target_embedding, p=2, dim=0)
    initial_distance = (1.0 - F.cosine_similarity(
        input_emb_norm.unsqueeze(0),
        target_emb_norm.unsqueeze(0)
    )).item()
    print(f"  Distance (input -> target): {initial_distance:.4f}")
    print()

    # Load TCN checkpoint
    print("Loading pretrained TCN and FiLM generator...")
    checkpoint = torch.load(args.tcn_checkpoint, weights_only=False, map_location=device)

    # Load TCN
    tcn = TCNMixer(
        in_channels=8,
        hidden_channels=checkpoint.get('hidden_channels', 16),
        num_blocks=checkpoint.get('num_blocks', 8),
        kernel_size=checkpoint.get('kernel_size', 5),
        causal=checkpoint.get('causal', False),
        use_film=True
    ).to(device)
    tcn.load_state_dict(checkpoint['tcn_state_dict'])
    tcn.eval()

    # Load FiLM generator
    film_input_dim = 2 * encoder_embed_dim
    film_generator = TCNFiLMGenerator(
        embed_dim=film_input_dim,
        num_blocks=checkpoint.get('num_blocks', 8),
        hidden_channels=checkpoint.get('hidden_channels', 16)
    ).to(device)
    film_generator.load_state_dict(checkpoint['film_generator_state_dict'])
    film_generator.eval()

    print(f"  TCN parameters: {sum(p.numel() for p in tcn.parameters()):,}")
    print(f"  FiLM generator parameters: {sum(p.numel() for p in film_generator.parameters()):,}")
    print()

    # Apply style transfer
    print("Applying style transfer...")
    result = apply_style_transfer(
        tcn=tcn,
        film_generator=film_generator,
        stems_input=input_stems,
        target_embedding=target_embedding,
        input_embedding=input_embedding,
        device=device
    )
    print("  Style transfer complete!")
    print()

    # Compute final embedding and distance
    print("Computing final distance...")
    transferred_mixture = result['processed_mixture']
    transferred_stems = result['processed_stems']

    final_embedding = compute_mixing_embedding(
        transferred_mixture,
        transferred_stems,
        mixing_model,
        feature_extractor,
        device,
        encoder_type=args.encoder_type
    )
    final_emb_norm = F.normalize(final_embedding, p=2, dim=0)
    final_distance = (1.0 - F.cosine_similarity(
        final_emb_norm.unsqueeze(0),
        target_emb_norm.unsqueeze(0)
    )).item()
    print(f"  Final distance: {final_distance:.4f}")
    print(f"  Improvement: {(initial_distance - final_distance) / initial_distance * 100:.1f}%")
    print()

    # Save results
    print("Saving results...")

    # Save transferred audio
    sf.write(str(output_dir / 'transferred_audio.wav'), transferred_mixture.numpy().T, 44100)

    # Save transferred stems
    transferred_stems_dir = output_dir / 'transferred_stems'
    transferred_stems_dir.mkdir(exist_ok=True)
    for stem_name, stem_audio in transferred_stems.items():
        sf.write(str(transferred_stems_dir / f'{stem_name}.wav'), stem_audio.numpy().T, 44100)

    # Save metadata
    metadata = {
        'encoder_type': args.encoder_type,
        'encoder_embed_dim': encoder_embed_dim,
        'input_audio': str(args.input_audio),
        'target_audio': str(args.target_audio),
        'tcn_checkpoint': str(args.tcn_checkpoint),
        'segment_duration': args.segment_duration,
        'segment_offset': args.segment_offset,
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'improvement': (initial_distance - final_distance) / initial_distance * 100
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Results saved to: {output_dir}")
    print(f"  Transferred audio: {output_dir / 'transferred_audio.wav'}")
    print()

    # Print summary
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Encoder: {args.encoder_type} ({encoder_embed_dim}-dim)")
    print(f"Initial distance: {initial_distance:.4f}")
    print(f"Final distance: {final_distance:.4f}")
    print(f"Improvement: {metadata['improvement']:.1f}%")
    print("="*80)


if __name__ == '__main__':
    main()
