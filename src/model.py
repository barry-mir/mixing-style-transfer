"""
Band-split CNN encoder for mixing style representation learning.

Adapted from SubSpectralNet architecture to process 8-channel input:
- 4 stems (vocals, bass, drums, other) × 2 stereo channels = 8 channels
- Processes sub-spectrograms with overlapping bands
- FiLM conditioning based on mixing features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


class MelSpectrogramPreprocessor(nn.Module):
    """
    Converts raw audio stems to 8-channel mel spectrogram.

    Input: Dict of 4 stems (vocals, bass, drums, other), each (B, 2, T)
    Output: 8-channel mel spectrogram (B, 8, n_mels, time_frames)
    """

    def __init__(self, sample_rate=44100, n_fft=1024, hop_length=256, n_mels=128):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

    def forward(self, stems_dict):
        """
        Convert stems to mel spectrogram.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (B, 2, T) for stereo

        Returns:
            mel_spec: 8-channel mel spectrogram (B, 8, n_mels, time_frames)
        """
        mel_specs = []

        for stem_name in ['vocals', 'bass', 'drums', 'other']:
            stem_audio = stems_dict[stem_name]  # (B, 2, T)

            # Compute mel spectrogram for both channels
            stem_mel = self.mel_transform(stem_audio)  # (B, 2, n_mels, time_frames)
            mel_specs.append(stem_mel)

        # Stack all stems: 4 stems × 2 channels = 8 channels
        mel_spec = torch.cat(mel_specs, dim=1)  # (B, 8, n_mels, time_frames)

        # Log scale
        mel_spec = torch.log(mel_spec + 1e-10)

        return mel_spec


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x, gamma, beta):
        """
        Apply FiLM: x' = gamma * x + beta

        Args:
            x: Input tensor (B, C, ...) where C is the modulated dimension
            gamma: Scale parameter (B, C)
            beta: Shift parameter (B, C)

        Returns:
            Modulated tensor of same shape as x
        """
        # Reshape gamma and beta to match x dimensions
        shape = [x.size(0), self.num_features] + [1] * (x.ndim - 2)
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)

        return gamma * x + beta


class SubSpectrogramCNN(nn.Module):
    """
    CNN module for processing a single sub-spectrogram band.

    This is one branch of the band-split architecture.
    Returns 2D features (no flattening) for temporal attention pooling.
    """

    def __init__(self, split_size, channels, out_channels=64):
        super().__init__()
        self.split_size = split_size
        self.channels = channels
        self.out_channels = out_channels

        sub_size = max(1, split_size // 10)  # Vertical pooling size

        # First conv layer
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.film1 = FiLMLayer(32)
        self.pool1 = nn.MaxPool2d((sub_size, 5))
        self.dropout1 = nn.Dropout(0.3)

        # Second conv layer
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.film2 = FiLMLayer(out_channels)
        self.pool2 = nn.MaxPool2d((4, 4))  # Less aggressive time pooling
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x, gamma1=None, beta1=None, gamma2=None, beta2=None):
        """
        Forward pass through sub-spectrogram CNN.

        Args:
            x: Input sub-spectrogram (B, C, H, W)
            gamma1, beta1: FiLM params for first conv layer
            gamma2, beta2: FiLM params for second conv layer

        Returns:
            features: 2D features (B, out_channels, H', W') where W' is time dimension
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        if gamma1 is not None and beta1 is not None:
            x = self.film1(x, gamma1, beta1)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        if gamma2 is not None and beta2 is not None:
            x = self.film2(x, gamma2, beta2)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x  # (B, out_channels, H', W')


class AttentionPooling(nn.Module):
    """
    Attention pooling layer for temporal aggregation.

    Learns to weight different time frames based on their importance
    for mixing style representation.
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Attention mechanism (operates on each time frame)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        """
        Apply attention pooling across time dimension.

        Args:
            x: Features (B, C, T) where T is time dimension

        Returns:
            pooled: Attention-pooled features (B, output_dim)
        """
        # Transpose to (B, T, C) for attention computation
        x = x.transpose(1, 2)  # (B, T, C)

        # Compute attention scores for each time frame
        attn_scores = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)

        # Weighted sum across time dimension
        weighted_features = x * attn_weights  # (B, T, C)
        pooled_features = weighted_features.sum(dim=1)  # (B, C)

        # Project to output dimension
        output = self.projection(pooled_features)  # (B, output_dim)

        return output


class BandSplitEncoder(nn.Module):
    """
    Band-split CNN encoder for 8-channel stem input.

    Architecture:
    1. Split mel-spectrogram into overlapping sub-bands
    2. Process each sub-band with dedicated CNN (outputs 2D features)
    3. Concatenate sub-band features along frequency dimension
    4. Attention pooling over time dimension
    5. Projection to final embedding
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        split_size=20,
        overlap=10,
        channels=8,
        embed_dim=768,
        cnn_out_channels=64
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split_size = split_size
        self.overlap = overlap
        self.channels = channels
        self.cnn_out_channels = cnn_out_channels

        # Mel spectrogram preprocessor
        self.mel_preprocessor = MelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # Calculate number of sub-bands
        self.n_subbands = 0
        i = 0
        while overlap * i <= n_mels - split_size:
            self.n_subbands += 1
            i += 1

        # Create sub-spectrogram CNNs
        self.subnet_cnns = nn.ModuleList([
            SubSpectrogramCNN(split_size, channels, out_channels=cnn_out_channels)
            for _ in range(self.n_subbands)
        ])

        # Calculate the total feature dimension after concatenation
        # Need to compute output shape from a dummy forward pass with 10s audio
        dummy_audio_length = int(10.0 * sample_rate)  # 10 seconds
        dummy_time_frames = (dummy_audio_length // hop_length) + 1

        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, split_size, dummy_time_frames)
            dummy_output = self.subnet_cnns[0](dummy_input)
            self.freq_dim = dummy_output.shape[2]  # Height (frequency)
            self.time_dim = dummy_output.shape[3]  # Width (time)

        # Total channels after concatenating all sub-bands
        total_channels = cnn_out_channels * self.n_subbands * self.freq_dim

        # Attention pooling layer (pools across time dimension)
        self.attention_pooling = AttentionPooling(
            input_dim=total_channels,
            hidden_dim=256,
            output_dim=embed_dim
        )

    def forward(self, stems_dict, film_params=None):
        """
        Forward pass through band-split encoder.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (B, 2, T) for stereo audio
            film_params: Optional dict with FiLM parameters from mixing features

        Returns:
            embedding: Track-level representation (B, embed_dim)
        """
        # Convert stems to mel spectrogram
        x = self.mel_preprocessor(stems_dict)  # (B, 8, n_mels, time_frames)

        # Debug: Check mel spectrogram
        if torch.isnan(x).any():
            print(f"[BandSplitEncoder] NaN in mel spectrogram after preprocessing!")
            print(f"  NaN count: {torch.isnan(x).sum()}/{x.numel()}")

        batch_size = x.size(0)

        # Pre-allocate concatenated tensor to avoid storing intermediate list
        # This reduces memory usage significantly
        # First pass: get dimensions from first sub-band
        start_idx = 0
        end_idx = self.split_size
        sub_band = x[:, :, start_idx:end_idx, :]

        # Get FiLM params if provided
        if film_params is not None:
            gamma1 = film_params.get(f'gamma1_0', None)
            beta1 = film_params.get(f'beta1_0', None)
            gamma2 = film_params.get(f'gamma2_0', None)
            beta2 = film_params.get(f'beta2_0', None)
        else:
            gamma1 = beta1 = gamma2 = beta2 = None

        first_features = self.subnet_cnns[0](sub_band, gamma1, beta1, gamma2, beta2)
        _, out_channels, freq_dim, time_dim = first_features.shape

        # Pre-allocate concatenated tensor
        concat_features = torch.empty(
            batch_size,
            self.n_subbands * out_channels,
            freq_dim,
            time_dim,
            dtype=first_features.dtype,
            device=first_features.device
        )

        # Fill in first sub-band
        concat_features[:, :out_channels, :, :] = first_features

        # Process remaining sub-bands directly into pre-allocated tensor
        for i in range(1, self.n_subbands):
            # Extract sub-band
            start_idx = i * self.overlap
            end_idx = start_idx + self.split_size
            sub_band = x[:, :, start_idx:end_idx, :]

            # Get FiLM params if provided
            if film_params is not None:
                gamma1 = film_params.get(f'gamma1_{i}', None)
                beta1 = film_params.get(f'beta1_{i}', None)
                gamma2 = film_params.get(f'gamma2_{i}', None)
                beta2 = film_params.get(f'beta2_{i}', None)
            else:
                gamma1 = beta1 = gamma2 = beta2 = None

            # Process sub-band and write directly to concat tensor
            features = self.subnet_cnns[i](sub_band, gamma1, beta1, gamma2, beta2)
            concat_features[:, i*out_channels:(i+1)*out_channels, :, :] = features

        # Flatten frequency and channel dimensions, keep time separate
        # (B, C, F, T) -> (B, C*F, T)
        batch_size, channels, freq, time = concat_features.shape
        flattened_features = concat_features.view(batch_size, channels * freq, time)

        # Debug: Check before attention pooling
        if torch.isnan(flattened_features).any():
            print(f"[BandSplitEncoder] NaN in flattened_features before attention pooling!")
            print(f"  NaN count: {torch.isnan(flattened_features).sum()}/{flattened_features.numel()}")

        # Apply attention pooling over time dimension
        embedding = self.attention_pooling(flattened_features)  # (B, embed_dim)

        # Debug: Check after attention pooling
        if torch.isnan(embedding).any():
            print(f"[BandSplitEncoder] NaN in embedding after attention pooling!")
            print(f"  NaN count: {torch.isnan(embedding).sum()}/{embedding.numel()}")

        return embedding


class MixingFeatureEncoder(nn.Module):
    """
    MLP for encoding mixing features and generating FiLM parameters.
    """

    def __init__(self, feature_dim, n_subbands, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_subbands = n_subbands

        # Feature embedding MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # FiLM parameter generators (for each sub-band CNN layer)
        # Each SubSpectrogramCNN has 2 conv layers with 32 and 64 channels
        film_output_dim = n_subbands * (32 + 32 + 64 + 64)  # gamma1, beta1, gamma2, beta2

        self.film_head = nn.Linear(hidden_dim, film_output_dim)

    def forward(self, features):
        """
        Generate FiLM parameters from mixing features.

        Args:
            features: Mixing features (B, feature_dim)

        Returns:
            film_params: Dict of FiLM parameters for each sub-band
        """
        # Debug: Check input features
        if torch.isnan(features).any():
            print(f"[MixingFeatureEncoder] NaN in input features!")
        if torch.isinf(features).any():
            print(f"[MixingFeatureEncoder] Inf in input features!")
        if (torch.abs(features) > 1000).any():
            print(f"[MixingFeatureEncoder] Extreme values in features! Max: {features.abs().max()}")

        # Embed features
        h = self.feature_mlp(features)  # (B, hidden_dim)

        # Debug: Check after MLP
        if torch.isnan(h).any():
            print(f"[MixingFeatureEncoder] NaN after feature_mlp!")
            print(f"  Input features stats: min={features.min()}, max={features.max()}, mean={features.mean()}")
        if torch.isinf(h).any():
            print(f"[MixingFeatureEncoder] Inf after feature_mlp!")

        # Generate FiLM params
        film_flat = self.film_head(h)  # (B, film_output_dim)

        # Debug: Check after film_head
        if torch.isnan(film_flat).any():
            print(f"[MixingFeatureEncoder] NaN after film_head!")
        if torch.isinf(film_flat).any():
            print(f"[MixingFeatureEncoder] Inf after film_head!")

        # Parse into individual parameters
        film_params = {}
        idx = 0

        for i in range(self.n_subbands):
            # gamma1, beta1 for first conv (32 channels)
            film_params[f'gamma1_{i}'] = film_flat[:, idx:idx+32]
            idx += 32
            film_params[f'beta1_{i}'] = film_flat[:, idx:idx+32]
            idx += 32

            # gamma2, beta2 for second conv (64 channels)
            film_params[f'gamma2_{i}'] = film_flat[:, idx:idx+64]
            idx += 64
            film_params[f'beta2_{i}'] = film_flat[:, idx:idx+64]
            idx += 64

        return film_params


class MixingStyleEncoder(nn.Module):
    """
    Complete mixing style encoder using ResNet50 with FiLM conditioning.

    Combines:
    1. ResNet50 audio encoder (memory efficient)
    2. FiLM conditioning based on mixing features
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        split_size=20,
        overlap=10,
        channels=8,
        embed_dim=768,
        feature_dim=256
    ):
        super().__init__()

        # Audio encoder (BandSplitCNN with FiLM conditioning and attention pooling)
        self.audio_encoder = BandSplitEncoder(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            split_size=split_size,
            overlap=overlap,
            channels=channels,
            embed_dim=embed_dim
        )

        # FiLM parameter generator for ResNet blocks
        self.film_encoder = MixingFeatureEncoder(
            feature_dim=feature_dim,
            n_subbands=self.audio_encoder.n_subbands
        )

    def forward(self, stems_dict, mixing_features):
        """
        Forward pass with FiLM conditioning.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (B, 2, T) for stereo audio
            mixing_features: Mixing features (B, feature_dim)

        Returns:
            embedding: Track-level representation (B, embed_dim)
        """
        # Debug: Check mixing features
        if torch.isnan(mixing_features).any():
            print(f"[MixingStyleEncoder] NaN in mixing_features input!")
            print(f"  NaN count: {torch.isnan(mixing_features).sum()}/{mixing_features.numel()}")

        # Generate FiLM parameters from mixing features
        film_params = self.film_encoder(mixing_features)

        # Debug: Check FiLM params
        for key, val in film_params.items():
            if torch.isnan(val).any():
                print(f"[MixingStyleEncoder] NaN in film_params[{key}]!")
                break

        # Encode audio with FiLM conditioning
        embedding = self.audio_encoder(stems_dict, film_params=film_params)  # (B, embed_dim)

        # Debug: Check final embedding
        if torch.isnan(embedding).any():
            print(f"[MixingStyleEncoder] NaN in final embedding!")
            print(f"  NaN count: {torch.isnan(embedding).sum()}/{embedding.numel()}")

        return embedding
