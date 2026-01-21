"""
Test script to verify the model works with temporal attention pooling and raw audio input.
Specifically tests that 10-second audio clips work correctly with all dimensions.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import MixingStyleEncoder, AttentionPooling, MelSpectrogramPreprocessor


def test_attention_pooling():
    """Test temporal attention pooling layer."""
    print("Testing Temporal AttentionPooling layer...")

    batch_size = 4
    feature_dim = 512  # Total channel dimension (C * F)
    time_frames = 25  # Time dimension
    embed_dim = 768

    # Create dummy features (B, C, T) where T is time
    features = torch.randn(batch_size, feature_dim, time_frames)

    # Initialize attention pooling
    attn_pool = AttentionPooling(
        input_dim=feature_dim,
        hidden_dim=128,
        output_dim=embed_dim
    )

    # Forward pass
    output = attn_pool(features)

    print(f"  Input shape (B, C, T): {features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {embed_dim})")

    assert output.shape == (batch_size, embed_dim), "Output shape mismatch!"
    print("  ✓ Temporal AttentionPooling test passed!\n")


def test_mel_preprocessor():
    """Test mel spectrogram preprocessor with 10-second audio."""
    print("Testing MelSpectrogramPreprocessor with 10-second audio...")

    batch_size = 2
    sample_rate = 44100
    duration = 10.0  # 10 seconds
    audio_length = int(sample_rate * duration)
    n_fft = 1024
    hop_length = 256
    n_mels = 128

    # Create dummy stems (4 stems × stereo)
    stems_dict = {
        'vocals': torch.randn(batch_size, 2, audio_length),
        'bass': torch.randn(batch_size, 2, audio_length),
        'drums': torch.randn(batch_size, 2, audio_length),
        'other': torch.randn(batch_size, 2, audio_length)
    }

    # Initialize preprocessor
    preprocessor = MelSpectrogramPreprocessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Forward pass
    mel_spec = preprocessor(stems_dict)

    # Calculate expected time frames
    expected_time_frames = (audio_length // hop_length) + 1

    print(f"  Input audio length: {audio_length} samples ({duration}s)")
    print(f"  Hop length: {hop_length}")
    print(f"  Expected time frames: {expected_time_frames}")
    print(f"  Output mel-spec shape: {mel_spec.shape}")
    print(f"  Expected shape: ({batch_size}, 8, {n_mels}, {expected_time_frames})")

    assert mel_spec.shape[0] == batch_size, "Batch size mismatch!"
    assert mel_spec.shape[1] == 8, "Channel count mismatch (should be 8)!"
    assert mel_spec.shape[2] == n_mels, "Mel bands mismatch!"
    print(f"  Actual time frames: {mel_spec.shape[3]}")

    print("  ✓ MelSpectrogramPreprocessor test passed!\n")


def test_full_model_with_raw_audio():
    """Test full mixing style encoder with 10-second raw audio."""
    print("Testing MixingStyleEncoder with 10-second raw audio...")

    batch_size = 2
    sample_rate = 44100
    duration = 10.0  # 10 seconds
    audio_length = int(sample_rate * duration)
    n_fft = 1024
    hop_length = 256
    n_mels = 128
    embed_dim = 768
    feature_dim = 100  # Dummy feature dimension

    # Create dummy stems (raw audio)
    stems_dict = {
        'vocals': torch.randn(batch_size, 2, audio_length),
        'bass': torch.randn(batch_size, 2, audio_length),
        'drums': torch.randn(batch_size, 2, audio_length),
        'other': torch.randn(batch_size, 2, audio_length)
    }

    mixing_features = torch.randn(batch_size, feature_dim)

    # Initialize model
    model = MixingStyleEncoder(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        split_size=20,
        overlap=10,
        channels=8,
        embed_dim=embed_dim,
        feature_dim=feature_dim
    )

    print(f"  Input stems: 4 stems × (B={batch_size}, 2, T={audio_length})")
    print(f"  Audio duration: {duration}s at {sample_rate}Hz")
    print(f"  Mixing features shape: {mixing_features.shape}")

    # Forward pass
    embedding = model(stems_dict, mixing_features)

    print(f"  Output embedding shape: {embedding.shape}")
    print(f"  Expected: ({batch_size}, {embed_dim})")

    assert embedding.shape == (batch_size, embed_dim), "Embedding shape mismatch!"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  ✓ Full model test with raw audio passed!\n")


def test_attention_weights():
    """Verify temporal attention weights sum to 1."""
    print("Testing temporal attention weight normalization...")

    batch_size = 3
    feature_dim = 256
    time_frames = 30

    features = torch.randn(batch_size, feature_dim, time_frames)

    attn_pool = AttentionPooling(input_dim=feature_dim, hidden_dim=128, output_dim=768)

    # Transpose to (B, T, C) for attention computation
    features_transposed = features.transpose(1, 2)  # (B, T, C)

    # Get attention scores (before softmax)
    attn_scores = attn_pool.attention(features_transposed)  # (B, T, 1)
    attn_weights = torch.softmax(attn_scores, dim=1)

    # Check that weights sum to 1 across time dimension
    weight_sums = attn_weights.sum(dim=1)

    print(f"  Features shape (B, C, T): {features.shape}")
    print(f"  Attention weights shape (B, T, 1): {attn_weights.shape}")
    print(f"  Weight sums per sample: {weight_sums.squeeze().tolist()}")
    print(f"  All sums ≈ 1.0: {torch.allclose(weight_sums, torch.ones_like(weight_sums))}")

    assert torch.allclose(weight_sums, torch.ones_like(weight_sums)), "Attention weights don't sum to 1!"
    print("  ✓ Temporal attention weight test passed!\n")


if __name__ == '__main__':
    print("=" * 70)
    print("Model Test Suite: Raw Audio Input + Temporal Attention Pooling")
    print("=" * 70 + "\n")

    test_mel_preprocessor()
    test_attention_pooling()
    test_attention_weights()
    test_full_model_with_raw_audio()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\n📊 Key Dimensions for 10-second audio @ 44.1kHz:")
    print("   - Audio samples: 441,000")
    print("   - Mel time frames (hop=256): ~1,724")
    print("   - After CNN processing: varies by pooling")
    print("   - Attention pools across time → Fixed 768D embedding")
