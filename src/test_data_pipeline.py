"""
Test data pipeline and export audio examples for verification.

This script:
1. Loads the FMA dataset with SCNet separation
2. Gets a sample triplet (anchor, positive, negative)
3. Exports audio files for listening
4. Shows augmentation details

Run with: python test_data_pipeline.py
"""

import os
import sys
import torch
import soundfile as sf
import numpy as np

# Add paths for imports
sys.path.insert(0, 'src')
sys.path.insert(0, 'Music-Source-Separation-Training')

from data import FMAContrastiveDataset, SCNetSeparator


def export_mixture_only(stems_dict, output_dir, prefix):
    """Export only the mixture (sum of all stems) to audio file."""
    os.makedirs(output_dir, exist_ok=True)

    # Export mixture (sum of all stems)
    mixture = sum(stems_dict.values())
    mixture_np = mixture.cpu().numpy().T  # (T, 2) for soundfile

    output_path = os.path.join(output_dir, f"{prefix}_mixture.wav")
    sf.write(output_path, mixture_np, 44100)
    print(f"  ✓ Exported: {output_path}")

    return mixture


def log_detailed_features(mixing_features, label, log_file):
    """Log detailed mixing features to file."""
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"{label} - Detailed Mixing Features\n")
        f.write(f"{'='*80}\n\n")

        # Convert to numpy for easier handling
        features = mixing_features.cpu().numpy()

        # Parse feature vector (based on MixingFeatureExtractor)
        # Each stem has: dynamics(6) + spectral(5) + stereo(3) + rel_loudness(1) = 15 features
        # Plus masking(4) at the end
        # Total: 4 stems * 15 + 4 = 64 features (approximate)

        idx = 0
        for stem_name in ['vocals', 'bass', 'drums', 'other']:
            f.write(f"\n{stem_name.upper()}:\n")
            f.write(f"  Dynamics:\n")
            f.write(f"    RMS L/R: {features[idx]:.4f}, {features[idx+1]:.4f}\n")
            f.write(f"    Crest Factor L/R: {features[idx+2]:.2f}, {features[idx+3]:.2f} dB\n")
            f.write(f"    Loudness: {features[idx+4]:.2f}, {features[idx+5]:.2f} LUFS\n")
            idx += 6

            f.write(f"  Spectral:\n")
            f.write(f"    Low Energy: {features[idx]:.2f} dB\n")
            f.write(f"    Mid Energy: {features[idx+1]:.2f} dB\n")
            f.write(f"    High Energy: {features[idx+2]:.2f} dB\n")
            f.write(f"    Spectral Tilt: {features[idx+3]:.4f}\n")
            f.write(f"    Spectral Flatness: {features[idx+4]:.4f}\n")
            idx += 5

            f.write(f"  Stereo:\n")
            f.write(f"    ILD: {features[idx]:.2f} dB\n")
            f.write(f"    Correlation: {features[idx+1]:.4f}\n")
            f.write(f"    MSR: {features[idx+2]:.4f}\n")
            idx += 3

            f.write(f"  Relative Loudness: {features[idx]:.2f} dB\n")
            idx += 1

        if idx < len(features) - 4:
            # Skip to masking features (last 4)
            idx = len(features) - 4

        f.write(f"\nINTER-STEM MASKING:\n")
        for i, stem_name in enumerate(['vocals', 'bass', 'drums', 'other']):
            if idx + i < len(features):
                f.write(f"  {stem_name}: {features[idx + i]:.4f}\n")

        f.write(f"\n")


def analyze_augmentation_differences(anchor_stems, negative_stems, log_file):
    """Analyze differences between anchor and negative (augmented) stems."""
    with open(log_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("Augmentation Analysis: Anchor vs Negative\n")
        f.write("="*80 + "\n\n")

        for stem_name in ['vocals', 'bass', 'drums', 'other']:
            anchor = anchor_stems[stem_name]
            negative = negative_stems[stem_name]

            # RMS level difference
            anchor_rms = torch.sqrt(torch.mean(anchor ** 2))
            negative_rms = torch.sqrt(torch.mean(negative ** 2))
            rms_diff_db = 20 * torch.log10(negative_rms / (anchor_rms + 1e-8))

            # Peak difference
            anchor_peak = torch.max(torch.abs(anchor))
            negative_peak = torch.max(torch.abs(negative))
            peak_diff_db = 20 * torch.log10(negative_peak / (anchor_peak + 1e-8))

            # Spectral centroid difference (simplified)
            anchor_fft = torch.fft.rfft(anchor[0])  # Use left channel
            negative_fft = torch.fft.rfft(negative[0])

            anchor_mag = torch.abs(anchor_fft)
            negative_mag = torch.abs(negative_fft)

            freqs = torch.arange(len(anchor_mag))
            anchor_centroid = torch.sum(freqs * anchor_mag) / (torch.sum(anchor_mag) + 1e-8)
            negative_centroid = torch.sum(freqs * negative_mag) / (torch.sum(negative_mag) + 1e-8)
            centroid_diff = negative_centroid - anchor_centroid

            f.write(f"{stem_name.upper()}:\n")
            f.write(f"  RMS difference: {rms_diff_db:.2f} dB\n")
            f.write(f"  Peak difference: {peak_diff_db:.2f} dB\n")
            f.write(f"  Spectral centroid shift: {centroid_diff:.1f} bins\n\n")

    print(f"✓ Augmentation analysis saved to {log_file}")


def test_dataset_sample():
    """Test dataset and export audio examples."""
    print("="*70)
    print("Testing Data Pipeline with Real FMA Data")
    print("="*70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Initialize SCNet separator
    print("\nInitializing SCNet separator...")
    scnet_separator = SCNetSeparator(
        model_path='../Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt',
        config_path='../Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml',
        device=device
    )
    print("✓ SCNet loaded")

    # Create dataset
    print("\nCreating FMA dataset...")
    dataset = FMAContrastiveDataset(
        data_path='/nas/FMA/fma_full/',
        scnet_separator=scnet_separator,
        clip_duration=10.0,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        augment_prob=0.5,
        gain_range=9.0,
        min_audio_duration=25.0
    )
    print(f"✓ Dataset created with {len(dataset)} samples")

    # Get a sample
    print("\n" + "="*70)
    print("Loading sample triplet (anchor, positive, negative)...")
    print("="*70)

    sample_idx = 10  # First sample
    stems_batch, mixing_features, labels = dataset[sample_idx]

    # Unpack triplet
    anchor_stems = {k: v[0] for k, v in stems_batch.items()}  # First in triplet
    positive_stems = {k: v[1] for k, v in stems_batch.items()}  # Second in triplet
    negative_stems = {k: v[2] for k, v in stems_batch.items()}  # Third in triplet

    print(f"\nAudio shape: {anchor_stems['vocals'].shape}")
    print(f"Duration: {anchor_stems['vocals'].shape[1] / 44100:.2f} seconds")
    print(f"Mixing features shape: {mixing_features.shape}")

    # Export audio files and create log
    output_dir = "data_pipeline_test"
    log_file = os.path.join(output_dir, "mixing_features.log")

    # Clear previous log
    if os.path.exists(log_file):
        os.remove(log_file)

    print(f"\n" + "="*70)
    print(f"Exporting audio files to: {output_dir}/")
    print(f"Logging detailed features to: {log_file}")
    print("="*70)

    print("\n🎵 ANCHOR (first 10s clip):")
    export_mixture_only(anchor_stems, output_dir, "anchor")
    log_detailed_features(mixing_features[0], "ANCHOR", log_file)

    print("\n🎵 POSITIVE (different 10s clip from same song):")
    print("   → Same mixing style, different content")
    export_mixture_only(positive_stems, output_dir, "positive")
    log_detailed_features(mixing_features[1], "POSITIVE", log_file)

    print("\n🎵 NEGATIVE (augmented anchor):")
    print("   → Same content, different mixing style")
    export_mixture_only(negative_stems, output_dir, "negative")
    log_detailed_features(mixing_features[2], "NEGATIVE", log_file)

    # Analyze augmentations
    print("\n📊 Analyzing augmentations...")
    analyze_augmentation_differences(anchor_stems, negative_stems, log_file)

    # Summary
    print("\n" + "="*70)
    print("Summary: What to Listen For")
    print("="*70)
    print("""
1. ANCHOR vs POSITIVE (Positive Pair):
   - Different musical content (different 10s clips)
   - Same mixing style (balance, EQ, dynamics, stereo)
   - Should sound like same production quality
   ✓ Listen: anchor_mixture.wav vs positive_mixture.wav

2. ANCHOR vs NEGATIVE (Negative Pair):
   - Same musical content (same notes/audio)
   - Different mixing style (altered balance, EQ, dynamics)
   - Augmentations applied: gain changes, EQ, compression, etc.
   ✓ Listen: anchor_mixture.wav vs negative_mixture.wav

3. DETAILED FEATURES:
   - Check mixing_features.log for complete analysis
   - Per-stem: dynamics, spectral, stereo, loudness
   - Inter-stem masking scores
   - Augmentation differences
    """)

    print("\n✅ Data pipeline test completed!")
    print(f"   📁 Audio: {os.path.abspath(output_dir)}/")
    print(f"   📄 Log: {os.path.abspath(log_file)}")


if __name__ == '__main__':
    test_dataset_sample()
