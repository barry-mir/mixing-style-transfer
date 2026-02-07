"""
Temporal Convolutional Network for differentiable mixing style transfer.

Architecture:
- Input: 8 channels (4 stems × stereo)
- Output: 8 channels (processed stems)
- Receptive field: ~2 seconds at 44.1kHz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with padding to maintain temporal dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class NonCausalConv1d(nn.Module):
    """
    Non-causal 1D convolution with symmetric padding to maintain temporal dimension.
    Used for non-causal TCN where future context is available.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # Symmetric padding for non-causal (can see both past and future)
        self.padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    TCN residual block with dilated convolutions.

    Architecture based on: https://github.com/jhtonyKoo/music_mixing_style_transfer
    Uses BatchNorm + LeakyReLU for stable training.

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        causal: If True, uses causal convolution (no future context).
                If False, uses non-causal (symmetric padding).
    """
    def __init__(self, channels, kernel_size, dilation, causal=False):
        super().__init__()
        ConvLayer = CausalConv1d if causal else NonCausalConv1d
        self.conv1 = ConvLayer(channels, channels, kernel_size, dilation)
        self.conv2 = ConvLayer(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x + residual, negative_slope=0.2)
        return x


class FiLMResidualBlock(nn.Module):
    """
    TCN residual block with FiLM (Feature-wise Linear Modulation) conditioning.

    Architecture based on: https://github.com/jhtonyKoo/music_mixing_style_transfer
    Uses BatchNorm + LeakyReLU + FiLM conditioning for style transfer.

    Applies FiLM after each conv+norm layer:
        output = gamma * normalized_features + beta

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        causal: If True, uses causal convolution (no future context).
                If False, uses non-causal (symmetric padding).
    """
    def __init__(self, channels, kernel_size, dilation, causal=False):
        super().__init__()
        ConvLayer = CausalConv1d if causal else NonCausalConv1d
        self.conv1 = ConvLayer(channels, channels, kernel_size, dilation)
        self.conv2 = ConvLayer(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.channels = channels

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        """
        Args:
            x: (B, channels, T)
            gamma1, beta1: FiLM parameters for first layer (B, channels)
            gamma2, beta2: FiLM parameters for second layer (B, channels)

        Returns:
            (B, channels, T)
        """
        residual = x

        # First conv + norm + FiLM
        h = self.conv1(x)
        h = self.norm1(h)
        # Apply FiLM: expand gamma/beta to match temporal dimension
        h = gamma1.unsqueeze(-1) * h + beta1.unsqueeze(-1)  # (B, C, 1) * (B, C, T)
        h = F.leaky_relu(h, negative_slope=0.2)

        # Second conv + norm + FiLM
        h = self.conv2(h)
        h = self.norm2(h)
        h = gamma2.unsqueeze(-1) * h + beta2.unsqueeze(-1)
        h = F.leaky_relu(h, negative_slope=0.2)

        # Residual connection
        return h + residual


class TCNFiLMGenerator(nn.Module):
    """
    Generates FiLM parameters from concatenated embeddings for TCN conditioning.

    Args:
        embed_dim: Dimension of concatenated embeddings (input_emb + target_emb)
        num_blocks: Number of TCN residual blocks (default: 14)
        hidden_channels: Number of channels in TCN hidden layers (default: 128)

    For each block, generates 4 sets of parameters:
        gamma1, beta1 (for first FiLM layer), gamma2, beta2 (for second FiLM layer)
    """
    def __init__(self, embed_dim=1536, num_blocks=14, hidden_channels=128):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels

        # Total parameters needed: num_blocks * 4 * hidden_channels
        # (4 = gamma1 + beta1 + gamma2 + beta2)
        output_dim = num_blocks * 4 * hidden_channels

        # Larger MLP for larger output dimension
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )

        # Initialize to small values for stable start
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, concat_embeddings):
        """
        Args:
            concat_embeddings: (B, embed_dim) concatenated input+target embeddings

        Returns:
            List of dicts, one per block:
            [
                {'gamma1': (B, C), 'beta1': (B, C), 'gamma2': (B, C), 'beta2': (B, C)},
                ...
            ]
        """
        B = concat_embeddings.shape[0]

        # Generate all FiLM parameters
        params = self.mlp(concat_embeddings)  # (B, num_blocks * 4 * hidden_channels)

        # Reshape to (B, num_blocks, 4, hidden_channels)
        params = params.view(B, self.num_blocks, 4, self.hidden_channels)

        # Organize as list of dicts for each block
        film_params = []
        for i in range(self.num_blocks):
            film_params.append({
                'gamma1': params[:, i, 0, :],  # (B, hidden_channels)
                'beta1': params[:, i, 1, :],
                'gamma2': params[:, i, 2, :],
                'beta2': params[:, i, 3, :]
            })

        return film_params


class TCNMixer(nn.Module):
    """
    TCN-based mixing processor for style transfer.

    Architecture based on: https://github.com/jhtonyKoo/music_mixing_style_transfer
    Adapted for multi-stem (4 stems × stereo = 8 channels) mixing.

    Args:
        in_channels: Number of input channels (default: 8 for 4 stems × stereo)
        hidden_channels: Hidden layer channels (default: 128, reference architecture)
        num_blocks: Number of residual blocks (default: 14, reference uses 14)
        kernel_size: Convolution kernel size (default: 15, reference uses 15)
        causal: Use causal convolution (default: False, reference is non-causal)
        use_film: Whether to use FiLM conditioning (default: False for backward compatibility)

    Receptive field calculation (non-causal):
        RF = 1 + sum(dilation_i * (kernel_size - 1) for each layer)

    With 14 blocks, dilations [1,2,4,...,8192], kernel_size=15:
        RF = 1 + (2^14 - 1) * 14 = 229,377 samples ≈ 5.2 seconds at 44.1kHz
    """
    def __init__(
        self,
        in_channels=8,
        hidden_channels=128,
        num_blocks=14,
        kernel_size=15,
        causal=False,
        use_film=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_film = use_film
        self.num_blocks = num_blocks
        self.causal = causal

        # Input projection
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            if use_film:
                self.blocks.append(FiLMResidualBlock(hidden_channels, kernel_size, dilation, causal=causal))
            else:
                self.blocks.append(ResidualBlock(hidden_channels, kernel_size, dilation, causal=causal))

        # Output projection
        self.output_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

        # Initialize output layer to near-identity (small weights + bias=0)
        # This ensures TCN starts close to identity function via residual connection
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.output_conv.bias)

        # Calculate receptive field
        self.receptive_field = 1 + sum(
            2 ** i * (kernel_size - 1) for i in range(num_blocks)
        )
        print(f"TCNMixer receptive field: {self.receptive_field} samples "
              f"({self.receptive_field / 44100:.3f} seconds at 44.1kHz)")
        print(f"TCNMixer mode: {'causal' if causal else 'non-causal'}")

    def forward(self, x, film_params=None):
        """
        Args:
            x: (B, 8, T) tensor of stems [vocals_L, vocals_R, bass_L, bass_R, ...]
            film_params: Optional list of dicts containing FiLM parameters for each block.
                        Each dict has keys: 'gamma1', 'beta1', 'gamma2', 'beta2'
                        Required if use_film=True, ignored otherwise.

        Returns:
            Processed stems (B, 8, T)
        """
        # Validate FiLM parameters
        if self.use_film:
            if film_params is None:
                raise ValueError("film_params must be provided when use_film=True")
            if len(film_params) != self.num_blocks:
                raise ValueError(f"Expected {self.num_blocks} FiLM parameter dicts, got {len(film_params)}")

        # Project to hidden dimension
        h = self.input_conv(x)

        # Apply TCN blocks
        if self.use_film:
            for i, block in enumerate(self.blocks):
                params = film_params[i]
                h = block(h, params['gamma1'], params['beta1'], params['gamma2'], params['beta2'])
        else:
            for block in self.blocks:
                h = block(h)

        # Project to output
        out = self.output_conv(h)

        # Residual connection with input
        out = out + x

        return out

    def process_stems_dict(self, stems_dict, film_params=None):
        """
        Convenience method to process stems dictionary.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is (2, T) tensor
            film_params: Optional FiLM parameters (required if use_film=True)

        Returns:
            Processed stems dict with same structure
        """
        # Stack stems: (8, T)
        stem_order = ['vocals', 'bass', 'drums', 'other']
        stems_list = []
        for stem_name in stem_order:
            stem = stems_dict[stem_name]  # (2, T)
            stems_list.append(stem)

        stacked = torch.cat(stems_list, dim=0)  # (8, T)
        stacked = stacked.unsqueeze(0)  # (1, 8, T)

        # Process
        processed = self.forward(stacked, film_params=film_params)  # (1, 8, T)
        processed = processed.squeeze(0)  # (8, T)

        # Unstack
        processed_dict = {}
        for i, stem_name in enumerate(stem_order):
            processed_dict[stem_name] = processed[i*2:(i+1)*2, :]  # (2, T)

        return processed_dict


def create_tcn_mixer(receptive_field_seconds=5.2, sample_rate=44100, use_film=False,
                     hidden_channels=8, kernel_size=15, causal=False):
    """
    Create TCN mixer with specified receptive field.

    Args:
        receptive_field_seconds: Desired receptive field in seconds (default: 5.2s, reference)
        sample_rate: Audio sample rate
        use_film: Whether to use FiLM conditioning (default: False)
        hidden_channels: Number of hidden channels (default: 128, reference architecture)
        kernel_size: Kernel size for convolutions (default: 15, reference architecture)
        causal: Use causal convolution (default: False, reference is non-causal)

    Returns:
        TCNMixer instance

    Note: With kernel_size=15 and 14 blocks (dilations 1,2,4,...,8192):
          RF = 1 + (2^14 - 1) * 14 = 229,377 samples ≈ 5.2s at 44.1kHz
    """
    target_rf_samples = int(receptive_field_seconds * sample_rate)

    # Calculate required number of blocks
    # RF = 1 + sum(2^i * (k-1)) for i in 0..n-1
    # With exponential dilation: RF ≈ (2^n - 1) * (kernel_size - 1) + 1
    # Solve for n: n = ceil(log2((RF - 1) / (k - 1) + 1))

    n = math.ceil(math.log2((target_rf_samples - 1) / (kernel_size - 1) + 1))

    # Clamp to reasonable range
    n = max(6, min(n, 16))  # 6-16 blocks

    print(f"Target receptive field: {target_rf_samples} samples ({receptive_field_seconds}s)")
    print(f"Using {n} TCN blocks with kernel_size={kernel_size}, hidden_channels={hidden_channels}")
    print(f"Causal mode: {causal}")
    print(f"FiLM conditioning: {'enabled' if use_film else 'disabled'}")

    return TCNMixer(
        in_channels=8,
        hidden_channels=hidden_channels,
        num_blocks=n,
        kernel_size=kernel_size,
        causal=causal,
        use_film=use_film
    )


if __name__ == "__main__":
    # Test TCN mixer
    tcn = create_tcn_mixer(receptive_field_seconds=2.0)

    # Test forward pass
    x = torch.randn(1, 8, 44100)  # 1 second of audio
    y = tcn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Receptive field: {tcn.receptive_field} samples")

    # Count parameters
    total_params = sum(p.numel() for p in tcn.parameters())
    print(f"Total parameters: {total_params:,}")
