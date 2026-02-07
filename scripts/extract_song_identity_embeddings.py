"""
Extract song identity embeddings using Discogs-VINet (CQTNet) model.

This script:
1. Loads the pretrained VINet model
2. For each track in the dataset:
   - Loads 4 stems and sums to mixture
   - Computes CQT spectrogram
   - Extracts 512-dim song identity embedding
3. Saves results to /ssd2/barry/fma_song_identity_embeddings.pt
"""

import os
import sys
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths
vinet_path = Path(__file__).parent.parent / 'Discogs-VINet'
sys.path.append(str(vinet_path))

# Import CQTNet directly
from model.nets.cqtnet import CQTNet

# Import mean_downsample_cqt directly from file to avoid loading full dataset package
import importlib.util
spec = importlib.util.spec_from_file_location("dataset_utils", vinet_path / 'model' / 'dataset' / 'dataset_utils.py')
dataset_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_utils)
mean_downsample_cqt = dataset_utils.mean_downsample_cqt


def load_vinet_model(checkpoint_path, device):
    """Load VINet (CQTNet) model from checkpoint."""
    print(f"Loading VINet model from: {checkpoint_path}")

    # Initialize model (from config.yaml)
    model = CQTNet(
        ch_in=40,           # CONV_CHANNEL
        ch_out=512,         # EMBEDDING_SIZE
        norm='bn',          # NORMALIZATION
        pool='adaptive_max', # POOLING
        l2_normalize=True,  # L2_NORMALIZE
        projection='linear'  # PROJECTION
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    print(f"Model loaded successfully")
    return model


def load_mixture_from_stems(track_dir, sr=22050):
    """
    Load 4 stems, sum to mixture, convert to mono, resample to 22050 Hz.

    Args:
        track_dir: Path to track directory with stems
        sr: Target sample rate (22050 for VINet)

    Returns:
        mixture: Mono mixture audio at 22050 Hz
    """
    stems = ['vocals', 'bass', 'drums', 'other']
    mixture = None

    for stem_name in stems:
        stem_path = os.path.join(track_dir, f"{stem_name}.mp3")
        if not os.path.exists(stem_path):
            raise FileNotFoundError(f"Stem not found: {stem_path}")

        # Load stem (librosa automatically converts to mono if mono=True)
        audio, original_sr = librosa.load(stem_path, sr=None, mono=False)

        # Convert stereo to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # Resample to target sr
        if original_sr != sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)

        # Accumulate mixture
        if mixture is None:
            mixture = audio
        else:
            mixture += audio

    return mixture


def compute_cqt(audio, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12):
    """
    Compute CQT spectrogram.

    Args:
        audio: Mono audio signal
        sr: Sample rate
        hop_length: Hop length for CQT
        n_bins: Number of frequency bins
        bins_per_octave: Bins per octave

    Returns:
        cqt_mag: CQT magnitude spectrogram, shape (n_bins, n_frames)
    """
    cqt = librosa.cqt(
        audio,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    )

    # Take magnitude
    cqt_mag = np.abs(cqt)

    return cqt_mag


def preprocess_cqt(cqt_mag, downsample_factor=20, context_length=7600):
    """
    Preprocess CQT: downsample, normalize, pad/crop to context length.

    Args:
        cqt_mag: CQT magnitude, shape (n_bins, n_frames)
        downsample_factor: Downsample factor (20 for VINet)
        context_length: Target length after processing (7600 for VINet)

    Returns:
        cqt_processed: Preprocessed CQT, shape (1, 1, n_bins, context_length)
    """
    # Transpose to (n_frames, n_bins) for mean_downsample_cqt
    cqt_transposed = cqt_mag.T  # (n_frames, n_bins)

    # Downsample by taking mean
    cqt_downsampled = mean_downsample_cqt(cqt_transposed, downsample_factor)

    # Transpose back to (n_bins, n_frames)
    cqt_downsampled = cqt_downsampled.T

    # Normalize (mean=0, std=1 per frequency bin)
    mean = cqt_downsampled.mean(axis=1, keepdims=True)
    std = cqt_downsampled.std(axis=1, keepdims=True) + 1e-8
    cqt_normalized = (cqt_downsampled - mean) / std

    # Pad or crop to context_length
    n_bins, n_frames = cqt_normalized.shape
    if n_frames < context_length:
        # Pad with zeros
        pad_width = ((0, 0), (0, context_length - n_frames))
        cqt_processed = np.pad(cqt_normalized, pad_width, mode='constant', constant_values=0)
    else:
        # Crop to context_length
        cqt_processed = cqt_normalized[:, :context_length]

    # Add batch and channel dimensions: (1, 1, n_bins, context_length)
    cqt_processed = cqt_processed[np.newaxis, np.newaxis, :, :]

    return cqt_processed


def extract_embedding(model, cqt_tensor, device):
    """
    Extract 512-dim song identity embedding from CQT.

    Args:
        model: VINet model
        cqt_tensor: CQT tensor, shape (1, 1, n_bins, context_length)
        device: torch device

    Returns:
        embedding: 512-dim L2-normalized embedding
    """
    with torch.no_grad():
        cqt_tensor = torch.from_numpy(cqt_tensor).float().to(device)
        embedding = model(cqt_tensor)  # (1, 512)

    return embedding.cpu()


def extract_all_embeddings(dataset_path, model, device, output_path):
    """
    Extract embeddings for all tracks in dataset.

    Args:
        dataset_path: Path to separated stems dataset
        model: VINet model
        device: torch device
        output_path: Path to save embeddings cache
    """
    # Find all valid track directories (must contain vocals.mp3)
    track_dirs = sorted([
        d for d in Path(dataset_path).iterdir()
        if d.is_dir() and (d / 'vocals.mp3').exists()
    ])

    print(f"Found {len(track_dirs)} tracks in {dataset_path}")

    # Extract embeddings
    embeddings = []
    track_paths = []
    failed_tracks = []

    for track_dir in tqdm(track_dirs, desc="Extracting embeddings"):
        try:
            # Load mixture from stems
            mixture = load_mixture_from_stems(str(track_dir), sr=22050)

            # Compute CQT
            cqt_mag = compute_cqt(mixture, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12)

            # Preprocess CQT
            cqt_processed = preprocess_cqt(cqt_mag, downsample_factor=20, context_length=7600)

            # Extract embedding
            embedding = extract_embedding(model, cqt_processed, device)

            # Store results
            embeddings.append(embedding)
            track_paths.append(str(track_dir))

        except Exception as e:
            print(f"\nError processing {track_dir.name}: {e}")
            failed_tracks.append(str(track_dir))
            continue

    # Stack embeddings
    embeddings = torch.cat(embeddings, dim=0)  # (N, 512)

    print(f"\nSuccessfully extracted {len(embeddings)} embeddings")
    print(f"Failed tracks: {len(failed_tracks)}")

    # Save cache
    cache = {
        'embeddings': embeddings,
        'track_paths': track_paths,
        'failed_tracks': failed_tracks
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(cache, output_path)
    print(f"Embeddings saved to: {output_path}")

    return cache


def main():
    parser = argparse.ArgumentParser(description='Extract song identity embeddings using VINet')
    parser.add_argument('--checkpoint', type=str,
                        default='Discogs-VINet/logs/checkpoints/Discogs-VINet-MIREX-full_set/model_checkpoint.pth',
                        help='Path to VINet checkpoint')
    parser.add_argument('--dataset_path', type=str,
                        default='/ssd2/barry/fma_large_separated/',
                        help='Path to separated stems dataset')
    parser.add_argument('--output_path', type=str,
                        default='/ssd2/barry/fma_large_separated/fma_song_identity_embeddings.pt',
                        help='Path to save embeddings cache')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test on first 10 tracks only')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_vinet_model(args.checkpoint, device)

    # Test on first 10 tracks if test mode
    if args.test_mode:
        print("\n=== TEST MODE: Processing first 10 tracks ===")
        track_dirs = sorted([
            d for d in Path(args.dataset_path).iterdir()
            if d.is_dir() and (d / 'vocals.mp3').exists()
        ])[:10]

        # Create temporary test dataset path
        test_output = args.output_path.replace('.pt', '_test.pt')

        embeddings = []
        track_paths = []

        for track_dir in tqdm(track_dirs, desc="Test extraction"):
            try:
                mixture = load_mixture_from_stems(str(track_dir), sr=22050)
                cqt_mag = compute_cqt(mixture, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12)
                cqt_processed = preprocess_cqt(cqt_mag, downsample_factor=20, context_length=7600)
                embedding = extract_embedding(model, cqt_processed, device)

                embeddings.append(embedding)
                track_paths.append(str(track_dir))

                print(f"  {track_dir.name}: embedding shape {embedding.shape}")

            except Exception as e:
                print(f"\n  Error: {e}")
                import traceback
                traceback.print_exc()

        embeddings = torch.cat(embeddings, dim=0)
        cache = {'embeddings': embeddings, 'track_paths': track_paths, 'failed_tracks': []}
        torch.save(cache, test_output)

        print(f"\nTest complete! Extracted {len(embeddings)} embeddings")
        print(f"Test cache saved to: {test_output}")

    else:
        # Extract embeddings for all tracks
        print("\n=== FULL EXTRACTION MODE ===")
        cache = extract_all_embeddings(args.dataset_path, model, device, args.output_path)

        print("\n=== EXTRACTION COMPLETE ===")
        print(f"Total embeddings: {cache['embeddings'].shape[0]}")
        print(f"Embedding dimension: {cache['embeddings'].shape[1]}")
        print(f"Output file: {args.output_path}")


if __name__ == '__main__':
    main()
